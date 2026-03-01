"""
J.A.S.O.N. Spider Crawler Protocol
Deep-Web Data Acquisition Engine: Real-time pricing, supplier data, and market trends
"""

import os
import json
import time
import hashlib
import logging
import asyncio
import re
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from urllib.parse import urlparse, urljoin, quote_plus
from pathlib import Path
import threading
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# HTTP & Scraping
try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    requests_available = True
except ImportError:
    requests_available = False

try:
    from bs4 import BeautifulSoup
    bs4_available = True
except ImportError:
    bs4_available = False

try:
    import aiohttp
    aiohttp_available = True
except ImportError:
    aiohttp_available = False

logger = logging.getLogger(__name__)


class SpiderCrawler:
    """
    Deep-Web Spider Crawler for J.A.R.V.I.S.
    Fetches live pricing, API documentation, supplier data, and market trends.
    """

    # User agent rotation for stealth
    USER_AGENTS = [
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",
    ]

    def __init__(self, config: dict = None):
        self.config = config or {}
        spider_config = self.config.get('spider', {})

        # Crawl settings
        self.max_depth = spider_config.get('max_depth', 3)
        self.max_pages = spider_config.get('max_pages', 100)
        self.rate_limit_seconds = spider_config.get('rate_limit_seconds', 1.5)
        self.timeout = spider_config.get('timeout', 15)
        self.respect_robots = spider_config.get('respect_robots', True)

        # Cache settings
        self.cache_dir = Path(os.path.dirname(__file__)) / '..' / '..' / 'crawl_data'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_ttl_hours = spider_config.get('cache_ttl_hours', 6)

        # Session management
        self._session = None
        self._ua_index = 0

        # Crawl state
        self.visited_urls = set()
        self.crawl_results = []
        self.crawl_stats = {
            'pages_crawled': 0,
            'data_points_extracted': 0,
            'errors': 0,
            'cache_hits': 0,
            'last_crawl_time': None,
            'active': False,
        }

        # Supplier endpoints (public search/filter APIs)
        self.supplier_endpoints = {
            'aliexpress': {
                'search_url': 'https://www.aliexpress.com/w/wholesale-{query}.html',
                'trending_url': 'https://www.aliexpress.com/popular.html',
                'category_url': 'https://www.aliexpress.com/category/{category_id}.html',
            },
            'dhgate': {
                'search_url': 'https://www.dhgate.com/wholesale/search.do?searchkey={query}',
            },
            'cjdropshipping': {
                'search_url': 'https://cjdropshipping.com/search.html?key={query}',
            },
            'temu': {
                'search_url': 'https://www.temu.com/search_result.html?search_key={query}',
            },
        }

        # Market trend sources
        self.trend_sources = {
            'google_trends': 'https://trends.google.com/trends/explore?q={query}',
            'amazon_movers': 'https://www.amazon.com/gp/movers-and-shakers/',
            'ebay_trending': 'https://www.ebay.com/deals',
        }

        logger.info("Spider Crawler initialized")

    def _get_session(self) -> requests.Session:
        """Get or create HTTP session with retry logic"""
        if not requests_available:
            raise RuntimeError("requests library not available")

        if self._session is None:
            self._session = requests.Session()
            self._session.verify = False  # Fix macOS SSL cert issues
            retry_strategy = Retry(
                total=3,
                backoff_factor=1,
                status_forcelist=[429, 500, 502, 503, 504],
            )
            adapter = HTTPAdapter(max_retries=retry_strategy)
            self._session.mount("https://", adapter)
            self._session.mount("http://", adapter)

        # Rotate user agent
        self._ua_index = (self._ua_index + 1) % len(self.USER_AGENTS)
        self._session.headers.update({
            'User-Agent': self.USER_AGENTS[self._ua_index],
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'DNT': '1',
            'Connection': 'keep-alive',
        })

        return self._session

    def _cache_key(self, url: str) -> str:
        """Generate cache key from URL"""
        return hashlib.md5(url.encode()).hexdigest()

    def _get_cached(self, url: str) -> Optional[Dict]:
        """Get cached crawl result if still valid"""
        cache_file = self.cache_dir / f"{self._cache_key(url)}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cached = json.load(f)
                cached_time = datetime.fromisoformat(cached.get('timestamp', '2000-01-01'))
                if datetime.now() - cached_time < timedelta(hours=self.cache_ttl_hours):
                    self.crawl_stats['cache_hits'] += 1
                    return cached
            except (json.JSONDecodeError, ValueError):
                pass
        return None

    def _save_cache(self, url: str, data: Dict):
        """Save crawl result to cache"""
        cache_file = self.cache_dir / f"{self._cache_key(url)}.json"
        data['timestamp'] = datetime.now().isoformat()
        data['source_url'] = url
        try:
            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Cache save failed: {e}")

    def crawl_url(self, url: str, extract_type: str = 'auto') -> Dict[str, Any]:
        """
        Crawl a single URL and extract structured data.
        
        extract_type: 'auto', 'products', 'pricing', 'text', 'links', 'api_docs'
        """
        # Check cache first
        cached = self._get_cached(url)
        if cached:
            logger.info(f"Cache hit for {url}")
            return cached

        if not requests_available:
            return {'error': 'requests library not available', 'url': url}

        try:
            session = self._get_session()
            time.sleep(self.rate_limit_seconds)  # Rate limiting

            response = session.get(url, timeout=self.timeout, allow_redirects=True)
            response.raise_for_status()

            self.visited_urls.add(url)
            self.crawl_stats['pages_crawled'] += 1

            # Parse content
            content_type = response.headers.get('Content-Type', '')

            if 'json' in content_type:
                result = self._extract_json_data(response.json(), url)
            elif 'html' in content_type and bs4_available:
                result = self._extract_html_data(response.text, url, extract_type)
            else:
                result = {
                    'url': url,
                    'content_type': content_type,
                    'raw_text': response.text[:5000],
                    'status_code': response.status_code,
                }

            result['crawl_timestamp'] = datetime.now().isoformat()
            result['response_time_ms'] = response.elapsed.total_seconds() * 1000

            # Cache the result
            self._save_cache(url, result)

            return result

        except requests.RequestException as e:
            self.crawl_stats['errors'] += 1
            logger.error(f"Crawl failed for {url}: {e}")
            return {'error': str(e), 'url': url}

    def _extract_html_data(self, html: str, url: str, extract_type: str) -> Dict[str, Any]:
        """Extract structured data from HTML"""
        soup = BeautifulSoup(html, 'html.parser')

        result = {
            'url': url,
            'title': soup.title.string if soup.title else '',
            'meta_description': '',
            'data_points': [],
            'links': [],
        }

        # Meta description
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc:
            result['meta_description'] = meta_desc.get('content', '')

        # Auto-detect extraction type
        if extract_type == 'auto':
            domain = urlparse(url).netloc.lower()
            if any(s in domain for s in ['aliexpress', 'dhgate', 'amazon', 'ebay', 'temu', 'cj']):
                extract_type = 'products'
            elif 'docs' in domain or 'api' in domain or 'developer' in domain:
                extract_type = 'api_docs'
            else:
                extract_type = 'text'

        if extract_type == 'products':
            result['products'] = self._extract_products(soup, url)
            self.crawl_stats['data_points_extracted'] += len(result['products'])

        elif extract_type == 'pricing':
            result['pricing'] = self._extract_pricing(soup)
            self.crawl_stats['data_points_extracted'] += len(result['pricing'])

        elif extract_type == 'api_docs':
            result['api_documentation'] = self._extract_api_docs(soup)

        elif extract_type == 'text':
            result['text_content'] = self._extract_text_content(soup)

        # Extract all links for deeper crawling
        for link in soup.find_all('a', href=True):
            href = link['href']
            absolute_url = urljoin(url, href)
            if absolute_url.startswith(('http://', 'https://')):
                result['links'].append({
                    'url': absolute_url,
                    'text': link.get_text(strip=True)[:100],
                })

        return result

    def _extract_products(self, soup: BeautifulSoup, base_url: str) -> List[Dict]:
        """Extract product data from e-commerce pages"""
        products = []

        # Generic product extraction patterns
        product_selectors = [
            # AliExpress patterns
            {'container': '[class*="product"]', 'name': '[class*="title"]', 'price': '[class*="price"]'},
            {'container': '[class*="item"]', 'name': 'h3, h4, .title', 'price': '.price, [class*="price"]'},
            # Generic patterns
            {'container': '.product-card, .product-item, .search-item', 'name': '.product-title, .item-title', 'price': '.product-price, .item-price'},
            # Schema.org structured data
            {'container': '[itemtype*="Product"]', 'name': '[itemprop="name"]', 'price': '[itemprop="price"]'},
        ]

        for selector in product_selectors:
            containers = soup.select(selector['container'])
            for container in containers[:50]:  # Limit to 50 products
                product = {}

                name_el = container.select_one(selector['name'])
                if name_el:
                    product['name'] = name_el.get_text(strip=True)[:200]

                price_el = container.select_one(selector['price'])
                if price_el:
                    price_text = price_el.get_text(strip=True)
                    product['price_raw'] = price_text
                    # Extract numeric price
                    price_match = re.search(r'[\$€£¥]?\s*([\d,]+\.?\d*)', price_text)
                    if price_match:
                        product['price'] = float(price_match.group(1).replace(',', ''))

                # Image
                img = container.find('img')
                if img:
                    product['image_url'] = img.get('src') or img.get('data-src', '')

                # Link
                link = container.find('a', href=True)
                if link:
                    product['product_url'] = urljoin(base_url, link['href'])

                # Rating
                rating_el = container.select_one('[class*="rating"], [class*="star"]')
                if rating_el:
                    rating_text = rating_el.get_text(strip=True)
                    rating_match = re.search(r'([\d.]+)', rating_text)
                    if rating_match:
                        product['rating'] = float(rating_match.group(1))

                # Orders/Sales
                orders_el = container.select_one('[class*="order"], [class*="sold"], [class*="sale"]')
                if orders_el:
                    product['orders_text'] = orders_el.get_text(strip=True)

                if product.get('name'):
                    products.append(product)

            if products:
                break  # Found products, no need to try other selectors

        # Also try JSON-LD structured data
        for script in soup.find_all('script', type='application/ld+json'):
            try:
                ld_data = json.loads(script.string)
                if isinstance(ld_data, dict) and ld_data.get('@type') == 'Product':
                    products.append({
                        'name': ld_data.get('name', ''),
                        'price': float(ld_data.get('offers', {}).get('price', 0)),
                        'currency': ld_data.get('offers', {}).get('priceCurrency', 'USD'),
                        'description': ld_data.get('description', '')[:300],
                        'image_url': ld_data.get('image', ''),
                        'source': 'json-ld',
                    })
                elif isinstance(ld_data, list):
                    for item in ld_data:
                        if isinstance(item, dict) and item.get('@type') == 'Product':
                            products.append({
                                'name': item.get('name', ''),
                                'price': float(item.get('offers', {}).get('price', 0)),
                                'description': item.get('description', '')[:300],
                                'source': 'json-ld',
                            })
            except (json.JSONDecodeError, ValueError, TypeError):
                pass

        return products

    def _extract_pricing(self, soup: BeautifulSoup) -> List[Dict]:
        """Extract pricing information from pages"""
        pricing = []

        # Look for pricing tables
        tables = soup.find_all('table')
        for table in tables:
            rows = table.find_all('tr')
            headers = []
            for row in rows:
                cells = row.find_all(['th', 'td'])
                cell_texts = [c.get_text(strip=True) for c in cells]

                if not headers:
                    headers = cell_texts
                else:
                    if len(cell_texts) == len(headers):
                        pricing.append(dict(zip(headers, cell_texts)))

        # Look for pricing cards
        price_cards = soup.select('[class*="pricing"], [class*="plan"], [class*="tier"]')
        for card in price_cards:
            plan = {
                'plan_name': '',
                'price': '',
                'features': [],
            }
            title = card.select_one('h2, h3, h4, [class*="title"], [class*="name"]')
            if title:
                plan['plan_name'] = title.get_text(strip=True)

            price = card.select_one('[class*="price"], [class*="cost"], [class*="amount"]')
            if price:
                plan['price'] = price.get_text(strip=True)

            features = card.select('li, [class*="feature"]')
            plan['features'] = [f.get_text(strip=True) for f in features[:20]]

            if plan['plan_name'] or plan['price']:
                pricing.append(plan)

        return pricing

    def _extract_api_docs(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract API documentation structure"""
        docs = {
            'endpoints': [],
            'authentication': '',
            'base_url': '',
            'sections': [],
        }

        # Look for API endpoints
        code_blocks = soup.find_all(['code', 'pre'])
        for block in code_blocks:
            text = block.get_text(strip=True)
            # HTTP method patterns
            endpoint_match = re.search(r'(GET|POST|PUT|DELETE|PATCH)\s+(/\S+)', text)
            if endpoint_match:
                docs['endpoints'].append({
                    'method': endpoint_match.group(1),
                    'path': endpoint_match.group(2),
                    'context': text[:200],
                })

        # Section headers
        for header in soup.find_all(['h1', 'h2', 'h3']):
            docs['sections'].append({
                'level': int(header.name[1]),
                'title': header.get_text(strip=True),
            })

        return docs

    def _extract_text_content(self, soup: BeautifulSoup) -> str:
        """Extract clean text content from page"""
        # Remove scripts and styles
        for tag in soup(['script', 'style', 'nav', 'footer', 'header']):
            tag.decompose()

        text = soup.get_text(separator='\n', strip=True)
        # Clean up excessive whitespace
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return '\n'.join(lines[:500])

    def _extract_json_data(self, data: Any, url: str) -> Dict[str, Any]:
        """Process JSON response data"""
        return {
            'url': url,
            'content_type': 'application/json',
            'data': data,
            'data_points': len(data) if isinstance(data, (list, dict)) else 1,
        }

    # ─── HIGH-LEVEL COMMANDS ──────────────────────────────────────

    def search_suppliers(self, query: str, suppliers: List[str] = None) -> Dict[str, Any]:
        """Search multiple supplier platforms for a product query"""
        if suppliers is None:
            suppliers = list(self.supplier_endpoints.keys())

        results = {
            'query': query,
            'timestamp': datetime.now().isoformat(),
            'suppliers': {},
            'total_products': 0,
        }

        self.crawl_stats['active'] = True

        for supplier in suppliers:
            if supplier in self.supplier_endpoints:
                endpoint = self.supplier_endpoints[supplier]
                search_url = endpoint['search_url'].format(query=quote_plus(query))

                logger.info(f"Searching {supplier} for: {query}")
                crawl_result = self.crawl_url(search_url, extract_type='products')

                results['suppliers'][supplier] = {
                    'url': search_url,
                    'products': crawl_result.get('products', []),
                    'total': len(crawl_result.get('products', [])),
                    'error': crawl_result.get('error'),
                }
                results['total_products'] += len(crawl_result.get('products', []))

        self.crawl_stats['active'] = False
        self.crawl_stats['last_crawl_time'] = datetime.now().isoformat()

        return results

    def fetch_market_trends(self, query: str) -> Dict[str, Any]:
        """Fetch market trend data for a product category"""
        trends = {
            'query': query,
            'timestamp': datetime.now().isoformat(),
            'sources': {},
        }

        self.crawl_stats['active'] = True

        for source_name, url_template in self.trend_sources.items():
            url = url_template.format(query=quote_plus(query))
            logger.info(f"Fetching trends from {source_name}")
            crawl_result = self.crawl_url(url)
            trends['sources'][source_name] = {
                'url': url,
                'data': crawl_result,
            }

        self.crawl_stats['active'] = False
        return trends

    def deep_crawl(self, start_url: str, max_depth: int = None, max_pages: int = None) -> Dict[str, Any]:
        """
        Deep crawl starting from a URL, following links up to max_depth.
        Returns aggregated data from all crawled pages.
        """
        max_depth = max_depth or self.max_depth
        max_pages = max_pages or self.max_pages

        results = {
            'start_url': start_url,
            'timestamp': datetime.now().isoformat(),
            'pages': [],
            'total_pages': 0,
            'total_products': 0,
            'all_links': [],
        }

        self.crawl_stats['active'] = True
        queue = [(start_url, 0)]  # (url, depth)
        visited = set()

        while queue and len(results['pages']) < max_pages:
            url, depth = queue.pop(0)

            if url in visited or depth > max_depth:
                continue

            visited.add(url)
            logger.info(f"Deep crawl [{depth}/{max_depth}]: {url}")

            page_data = self.crawl_url(url)
            results['pages'].append(page_data)
            results['total_pages'] += 1

            if 'products' in page_data:
                results['total_products'] += len(page_data['products'])

            # Queue links for deeper crawling
            if depth < max_depth:
                for link in page_data.get('links', []):
                    link_url = link['url']
                    # Stay on same domain
                    if urlparse(link_url).netloc == urlparse(start_url).netloc:
                        if link_url not in visited:
                            queue.append((link_url, depth + 1))

        self.crawl_stats['active'] = False
        self.crawl_stats['last_crawl_time'] = datetime.now().isoformat()
        results['all_links'] = list(visited)

        return results

    def generate_seo_analysis(self, url: str) -> Dict[str, Any]:
        """Generate SEO analysis for a given URL"""
        crawl_data = self.crawl_url(url, extract_type='text')

        seo = {
            'url': url,
            'timestamp': datetime.now().isoformat(),
            'score': 0,
            'checks': {},
            'recommendations': [],
        }

        if 'error' in crawl_data:
            seo['error'] = crawl_data['error']
            return seo

        # Title tag
        title = crawl_data.get('title', '')
        seo['checks']['title'] = {
            'value': title,
            'length': len(title),
            'pass': 30 <= len(title) <= 60,
        }
        if seo['checks']['title']['pass']:
            seo['score'] += 15
        else:
            seo['recommendations'].append(f"Title should be 30-60 chars (currently {len(title)})")

        # Meta description
        desc = crawl_data.get('meta_description', '')
        seo['checks']['meta_description'] = {
            'value': desc,
            'length': len(desc),
            'pass': 100 <= len(desc) <= 160,
        }
        if seo['checks']['meta_description']['pass']:
            seo['score'] += 15
        else:
            seo['recommendations'].append(f"Meta description should be 100-160 chars (currently {len(desc)})")

        # Links analysis
        links = crawl_data.get('links', [])
        internal = [l for l in links if urlparse(l['url']).netloc == urlparse(url).netloc]
        external = [l for l in links if urlparse(l['url']).netloc != urlparse(url).netloc]
        seo['checks']['links'] = {
            'total': len(links),
            'internal': len(internal),
            'external': len(external),
            'pass': len(internal) > 3,
        }
        if seo['checks']['links']['pass']:
            seo['score'] += 10

        # Content length
        content = crawl_data.get('text_content', '')
        word_count = len(content.split())
        seo['checks']['content_length'] = {
            'word_count': word_count,
            'pass': word_count > 300,
        }
        if seo['checks']['content_length']['pass']:
            seo['score'] += 15
        else:
            seo['recommendations'].append(f"Content should be 300+ words (currently {word_count})")

        # Response time
        response_time = crawl_data.get('response_time_ms', 0)
        seo['checks']['performance'] = {
            'response_time_ms': response_time,
            'pass': response_time < 2000,
        }
        if seo['checks']['performance']['pass']:
            seo['score'] += 10

        seo['score'] = min(seo['score'], 100)

        return seo

    def get_crawl_stats(self) -> Dict[str, Any]:
        """Get current crawl statistics"""
        return {
            **self.crawl_stats,
            'cache_size': len(list(self.cache_dir.glob('*.json'))),
            'visited_count': len(self.visited_urls),
        }

    def clear_cache(self):
        """Clear all cached crawl data"""
        for f in self.cache_dir.glob('*.json'):
            f.unlink()
        logger.info("Spider Crawler cache cleared")

    def export_results(self, filepath: str) -> bool:
        """Export all crawl results to a JSON file"""
        try:
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'stats': self.get_crawl_stats(),
                'results': self.crawl_results,
            }
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            return True
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return False
