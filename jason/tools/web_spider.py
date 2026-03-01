"""
J.A.R.V.I.S. Web Spider Crawler
Advanced web crawling system for data collection and research
"""

import asyncio
import logging
from typing import Dict, Any, List, Set, Optional, Callable
from urllib.parse import urlparse, urljoin
from urllib.robotparser import RobotFileParser
import time
import json
import os
from pathlib import Path
import hashlib
import re
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import aiohttp
import aiofiles
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

@dataclass
class CrawlJob:
    """Represents a crawling job configuration"""
    name: str
    start_urls: List[str]
    max_depth: int = 3
    max_pages: int = 1000
    delay: float = 1.0  # seconds between requests
    respect_robots: bool = True
    allowed_domains: Optional[List[str]] = None
    blocked_patterns: Optional[List[str]] = None
    data_extractors: Optional[List[Dict[str, Any]]] = None
    storage_path: str = "crawl_data"
    user_agent: str = "J.A.R.V.I.S. Spider/1.0"

@dataclass
class CrawlResult:
    """Result from crawling a single page"""
    url: str
    status_code: int
    title: str
    content: str
    links: List[str]
    extracted_data: Dict[str, Any]
    crawl_time: datetime
    depth: int

class URLFrontier:
    """Manages the URL queue with politeness and prioritization"""

    def __init__(self, delay: float = 1.0):
        self.queue: asyncio.Queue = asyncio.Queue()
        self.visited: Set[str] = set()
        self.last_visit: Dict[str, float] = {}  # domain -> timestamp
        self.delay = delay

    def normalize_url(self, url: str, base_url: str = "") -> str:
        """Normalize URL by removing fragments and resolving relative URLs"""
        if not url.startswith(('http://', 'https://')):
            url = urljoin(base_url, url)
        parsed = urlparse(url)
        # Remove fragments
        return parsed._replace(fragment="").geturl()

    async def add_url(self, url: str, priority: int = 0, depth: int = 0) -> bool:
        """Add URL to queue if not visited"""
        normalized = self.normalize_url(url)
        if normalized in self.visited:
            return False

        domain = urlparse(normalized).netloc
        current_time = time.time()

        # Check politeness delay
        if domain in self.last_visit:
            time_since_last = current_time - self.last_visit[domain]
            if time_since_last < self.delay:
                # Schedule for later with proper async handling
                delay_needed = self.delay - time_since_last
                asyncio.get_event_loop().create_task(self._schedule_add(normalized, priority, depth, delay_needed))
                return True

        await self._add_to_queue(normalized, priority, depth)
        return True

    async def _schedule_add(self, url: str, priority: int, depth: int, delay: float):
        """Schedule URL addition after delay"""
        await asyncio.sleep(delay)
        await self._add_to_queue(url, priority, depth)

    async def _add_to_queue(self, url: str, priority: int, depth: int):
        """Add URL to queue"""
        await self.queue.put((url, priority, depth))

    def add_url_sync(self, url: str, depth: int = 0) -> bool:
        """Add URL to queue synchronously"""
        normalized = self.normalize_url(url)
        if normalized in self.visited:
            return False

        # For initial URLs, add directly to queue
        self.queue.put_nowait((normalized, 0, depth))
        return True

    async def get_url(self) -> Optional[tuple]:
        """Get next URL from queue (blocking)"""
        try:
            return await self.queue.get()
        except asyncio.QueueEmpty:
            return None

    def mark_visited(self, url: str):
        """Mark URL as visited"""
        self.visited.add(url)
        domain = urlparse(url).netloc
        self.last_visit[domain] = time.time()

class PolitenessManager:
    """Manages crawling politeness (robots.txt, delays, etc.)"""

    def __init__(self, user_agent: str = "J.A.R.V.I.S. Spider/1.0"):
        self.user_agent = user_agent
        self.robots_cache: Dict[str, RobotFileParser] = {}
        self.domain_delays: Dict[str, float] = {}

    async def can_fetch(self, url: str) -> bool:
        """Check if URL can be fetched according to robots.txt"""
        try:
            parsed = urlparse(url)
            robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"

            if robots_url not in self.robots_cache:
                rp = RobotFileParser()
                rp.set_url(robots_url)
                try:
                    await asyncio.get_event_loop().run_in_executor(None, rp.read)
                    self.robots_cache[robots_url] = rp
                except Exception:
                    # If robots.txt can't be read, assume allowed
                    self.robots_cache[robots_url] = None

            rp = self.robots_cache[robots_url]
            if rp:
                return rp.can_fetch(self.user_agent, url)
            return True
        except Exception:
            return True

    def get_delay(self, domain: str) -> float:
        """Get crawl delay for domain"""
        return self.domain_delays.get(domain, 1.0)

    def set_delay(self, domain: str, delay: float):
        """Set crawl delay for domain"""
        self.domain_delays[domain] = delay

class DataExtractor:
    """Extracts structured data from HTML content"""

    def __init__(self, extractors: List[Dict[str, Any]]):
        self.extractors = extractors

    def extract(self, html: str, url: str) -> Dict[str, Any]:
        """Extract data using configured extractors"""
        soup = BeautifulSoup(html, 'html.parser')
        results = {}

        for extractor in self.extractors:
            name = extractor.get('name', 'unnamed')
            selector = extractor.get('selector')
            attr = extractor.get('attribute')
            pattern = extractor.get('pattern')

            if selector:
                elements = soup.select(selector)
                if attr:
                    values = [elem.get(attr, '') for elem in elements if elem.get(attr)]
                else:
                    values = [elem.get_text(strip=True) for elem in elements]

                if pattern:
                    import re
                    values = [v for v in values if re.search(pattern, v)]

                results[name] = values if len(values) > 1 else (values[0] if values else None)

        return results

class WebSpider:
    """Main web spider crawler class"""

    def __init__(self, job: CrawlJob):
        self.job = job
        self.frontier = URLFrontier(job.delay)
        self.politeness = PolitenessManager(job.user_agent)
        self.extractor = DataExtractor(job.data_extractors or [])
        self.results: List[CrawlResult] = []
        self.storage_path = Path(job.storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Initialize URL frontier with start URLs (synchronously)
        for url in job.start_urls:
            self.frontier.add_url_sync(url, depth=0)

    async def crawl(self, progress_callback: Optional[Callable] = None) -> List[CrawlResult]:
        """Main crawling method"""
        logger.info(f"Starting crawl job: {self.job.name}")

        async with aiohttp.ClientSession(
            headers={'User-Agent': self.job.user_agent},
            timeout=aiohttp.ClientTimeout(total=30),
            connector=aiohttp.TCPConnector(verify_ssl=False)
        ) as session:

            while len(self.results) < self.job.max_pages:
                url_data = await self.frontier.get_url()
                if not url_data:
                    break

                url, priority, depth = url_data

                if depth > self.job.max_depth:
                    continue

                # Check domain restrictions
                if self.job.allowed_domains:
                    domain = urlparse(url).netloc
                    if domain not in self.job.allowed_domains:
                        continue

                # Check blocked patterns
                if self.job.blocked_patterns:
                    if any(re.search(pattern, url) for pattern in self.job.blocked_patterns):
                        continue

                # Check robots.txt
                if self.job.respect_robots:
                    if not await self.politeness.can_fetch(url):
                        logger.info(f"Blocked by robots.txt: {url}")
                        continue

                # Crawl the page
                result = await self._crawl_page(session, url, depth)
                if result:
                    self.results.append(result)

                    # Extract new URLs
                    for link in result.links:
                        if len(self.results) < self.job.max_pages:
                            await self.frontier.add_url(link, depth=depth + 1)

                    # Save result
                    await self._save_result(result)

                if progress_callback:
                    progress_callback(len(self.results), self.job.max_pages)

        logger.info(f"Crawl completed: {len(self.results)} pages crawled")
        return self.results

    async def _crawl_page(self, session: aiohttp.ClientSession, url: str, depth: int) -> Optional[CrawlResult]:
        """Crawl a single page"""
        try:
            async with session.get(url) as response:
                if response.status != 200:
                    logger.warning(f"Failed to fetch {url}: {response.status}")
                    return None

                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')

                # Extract title
                title = soup.title.string if soup.title else ""

                # Extract links
                links = []
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    if not href.startswith(('http://', 'https://')):
                        href = urljoin(url, href)
                    if href.startswith(('http://', 'https://')):
                        links.append(href)

                # Extract data
                extracted_data = self.extractor.extract(html, url)

                # Mark as visited
                self.frontier.mark_visited(url)

                return CrawlResult(
                    url=url,
                    status_code=response.status,
                    title=title,
                    content=html,
                    links=links[:100],  # Limit links
                    extracted_data=extracted_data,
                    crawl_time=datetime.now(),
                    depth=depth
                )

        except Exception as e:
            logger.error(f"Error crawling {url}: {e}")
            return None

    async def _save_result(self, result: CrawlResult):
        """Save crawl result to storage"""
        try:
            # Create filename from URL hash
            url_hash = hashlib.md5(result.url.encode()).hexdigest()
            filename = f"{url_hash}.json"
            filepath = self.storage_path / filename

            async with aiofiles.open(filepath, 'w') as f:
                await f.write(json.dumps(asdict(result), default=str, indent=2))

        except Exception as e:
            logger.error(f"Error saving result for {result.url}: {e}")

class SpiderManager:
    """Manages multiple crawling jobs and provides easy API"""

    def __init__(self):
        self.active_jobs: Dict[str, WebSpider] = {}
        self.completed_jobs: Dict[str, List[CrawlResult]] = {}

    async def start_job(self, job: CrawlJob, progress_callback: Optional[Callable] = None) -> str:
        """Start a new crawling job"""
        spider = WebSpider(job)
        job_id = f"{job.name}_{int(time.time())}"

        self.active_jobs[job_id] = spider

        # Run crawl in background
        asyncio.create_task(self._run_job(job_id, spider, progress_callback))

        return job_id

    async def _run_job(self, job_id: str, spider: WebSpider, progress_callback: Optional[Callable]):
        """Run crawling job"""
        try:
            results = await spider.crawl(progress_callback)
            self.completed_jobs[job_id] = results
        finally:
            del self.active_jobs[job_id]

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get status of a job"""
        if job_id in self.active_jobs:
            spider = self.active_jobs[job_id]
            return {
                "status": "running",
                "pages_crawled": len(spider.results),
                "max_pages": spider.job.max_pages
            }
        elif job_id in self.completed_jobs:
            return {
                "status": "completed",
                "pages_crawled": len(self.completed_jobs[job_id])
            }
        else:
            return {"status": "not_found"}

    def get_job_results(self, job_id: str) -> Optional[List[CrawlResult]]:
        """Get results of a completed job"""
        return self.completed_jobs.get(job_id)

# Easy-to-use API functions
async def crawl_website(url: str, max_pages: int = 100, max_depth: int = 3) -> List[CrawlResult]:
    """Simple function to crawl a website"""
    job = CrawlJob(
        name="quick_crawl",
        start_urls=[url],
        max_pages=max_pages,
        max_depth=max_depth
    )

    spider = WebSpider(job)
    return await spider.crawl()

async def crawl_with_config(config: Dict[str, Any]) -> List[CrawlResult]:
    """Crawl with custom configuration"""
    job = CrawlJob(**config)
    spider = WebSpider(job)
    return await spider.crawl()

# Example usage
if __name__ == "__main__":
    async def main():
        # Simple crawl
        results = await crawl_website("https://example.com", max_pages=10)
        print(f"Crawled {len(results)} pages")

        # Custom crawl
        config = {
            "name": "tech_news",
            "start_urls": ["https://techcrunch.com"],
            "max_pages": 50,
            "max_depth": 2,
            "allowed_domains": ["techcrunch.com"],
            "data_extractors": [
                {
                    "name": "headlines",
                    "selector": "h2 a",
                    "attribute": "text"
                }
            ]
        }
        results = await crawl_with_config(config)
        print(f"Crawled {len(results)} pages with custom config")

    asyncio.run(main())
