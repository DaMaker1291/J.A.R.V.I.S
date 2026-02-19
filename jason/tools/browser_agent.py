"""
J.A.S.O.N. Browser Agent Tool
Concierge Module - Autonomous Web Interaction
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
import requests
import time
import random
import urllib.parse

from playwright.async_api import async_playwright, Browser, Page, Playwright

logger = logging.getLogger(__name__)

class BrowserAgent:
    """Autonomous browser agent using Playwright"""

    def __init__(self, headless: bool = True, captcha_api_key: Optional[str] = None):
        self.headless = headless
        self.captcha_api_key = captcha_api_key
        self.playwright: Optional[Playwright] = None
        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None
        self._screenshot_task: Optional[asyncio.Task] = None
        self._stop_screenshots = asyncio.Event()
        self.screenshot_interval_seconds = 30
        self.screenshot_dir = Path("screenshots")

    async def __aenter__(self):
        """Async context manager entry"""
        import tempfile
        import os
        self.user_data_dir = tempfile.mkdtemp()
        # Browser settings
        self.browser_args = [
            '--start-maximized', 
            '--no-sandbox', 
            '--disable-web-security', 
            '--disable-blink-features=AutomationControlled',
            '--disable-infobars',
            '--window-position=0,0',
            '--ignore-certificate-errors',
            '--disable-extensions',
            '--disable-http2' # Added to prevent ERR_HTTP2_PROTOCOL_ERROR
        ]
        
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch_persistent_context(
            self.user_data_dir, 
            headless=self.headless, 
            args=self.browser_args,
            user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'
        )
        
        # Stealth tactics
        await self.browser.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined
            });
            window.chrome = {
                runtime: {}
            };
        """)
        
        self.page = self.browser.pages[0] if self.browser.pages else await self.browser.new_page()
        await self.page.set_viewport_size({"width": 1920, "height": 1080})

        self._stop_screenshots = asyncio.Event()
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)
        self._screenshot_task = asyncio.create_task(self._auto_screenshot_loop())
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await asyncio.sleep(5)  # Keep browser open for user to see
        try:
            self._stop_screenshots.set()
            if self._screenshot_task:
                await asyncio.gather(self._screenshot_task, return_exceptions=True)
        finally:
            self._screenshot_task = None
        if self.page:
            await self.page.close()
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()

    async def _auto_screenshot_loop(self) -> None:
        if not self.page:
            return
        first_capture_done = False
        while not self._stop_screenshots.is_set():
            try:
                if self.page.url and self.page.url != "about:blank":
                    try:
                        await self.page.wait_for_load_state("domcontentloaded", timeout=5000)
                    except Exception:
                        pass
                    ts = int(time.time())
                    await self.take_screenshot(f"screenshot_auto_{ts}.png")
                    first_capture_done = True
            except Exception:
                pass
            try:
                timeout = self.screenshot_interval_seconds if first_capture_done else 1
                await asyncio.wait_for(self._stop_screenshots.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                continue

    def _normalize_screenshot_path(self, path: str) -> Path:
        p = Path(path)
        if p.is_absolute():
            return p
        if p.parent == Path('.'):
            return self.screenshot_dir / p
        return p

    async def navigate(self, url: str) -> bool:
        """Navigate to URL with advanced error handling and typo resilience

        Args:
            url: Target URL

        Returns:
            bool: True if navigation successful
        """
        if not self.page:
            return False

        # Basic URL correction (typo resilience)
        if not url.startswith('http'):
            url = 'https://' + url
        
        # Correct common TLD typos
        url = url.replace('.comm', '.com').replace('.nett', '.net').replace('.orgg', '.org')

        try:
            # Random delay before navigation
            await asyncio.sleep(random.uniform(1.0, 3.0))
            
            # Using 'commit' as primary for heavy sites, then waiting for specific elements
            await self.page.goto(url, wait_until="commit", timeout=60000)
            logger.info(f"Navigated (commit) to {url}")
            
            # Additional wait for dynamic content
            await asyncio.sleep(10)
            
            # Take screenshot to inspect navigation success
            await self.take_screenshot(f"screenshot_navigate_{int(time.time())}.png")
            return True
        except Exception as e:
            logger.error(f"Navigation failed to {url}: {e}")
            return False

    async def search_and_click(self, selector: str, text: Optional[str] = None, retries: int = 3) -> bool:
        """Search for element and click it with retries

        Args:
            selector: CSS selector
            text: Optional text to match
            retries: Number of retry attempts

        Returns:
            bool: True if element found and clicked
        """
        if not self.page:
            return False

        for attempt in range(retries):
            try:
                if text:
                    element = self.page.locator(selector).filter(has_text=text).first
                else:
                    element = self.page.locator(selector).first

                await element.click()
                logger.info(f"Clicked element: {selector}")
                # Take screenshot to inspect click result
                await self.take_screenshot(f"screenshot_click_{int(time.time())}.png")
                return True
            except Exception as e:
                logger.warning(f"Click attempt {attempt + 1} failed: {e}")
                if attempt < retries - 1:
                    await asyncio.sleep(1)  # Wait before retry
                else:
                    logger.error(f"Click failed after {retries} attempts: {e}")
                    return False

    async def fill_form(self, fields: Dict[str, str]) -> bool:
        """Fill form fields with human-like typing"""
        try:
            for selector, value in fields.items():
                element = self.page.locator(selector).first
                if await element.is_visible():
                    await element.click()
                    await asyncio.sleep(random.uniform(0.5, 1.0))  # Human delay
                    await self.page.keyboard.type(value, delay=random.randint(100, 200))  # Slow typing
                    await asyncio.sleep(random.uniform(0.3, 0.7))
                else:
                    # Try alternative selectors
                    alt_selectors = [
                        f'[placeholder*="{value.split()[0]}"]',
                        f'input[aria-label*="{value.split()[0]}"]',
                        f'input[name*="{selector.split("[")[1].split("]")[0].replace("name=", "").replace("placeholder*=", "")}"]'
                    ]
                    for alt in alt_selectors:
                        try:
                            alt_element = self.page.locator(alt).first
                            if await alt_element.is_visible():
                                await alt_element.click()
                                await asyncio.sleep(random.uniform(0.5, 1.0))
                                await self.page.keyboard.type(value, delay=random.randint(100, 200))
                                await asyncio.sleep(random.uniform(0.3, 0.7))
                                break
                        except:
                            continue
            # Take screenshot after form filling
            await self.take_screenshot(f"screenshot_form_fill_{int(time.time())}.png")
            return True
        except Exception as e:
            print(f"Form fill error: {e}")
            return False

    async def wait_for_element(self, selector: str, timeout: int = 10000) -> bool:
        """Wait for element to appear

        Args:
            selector: CSS selector
            timeout: Timeout in ms

        Returns:
            bool: True if element found
        """
        if not self.page:
            return False

        try:
            await self.page.wait_for_selector(selector, timeout=timeout)
            return True
        except Exception:
            return False

    async def get_text(self, selector: str) -> Optional[str]:
        """Get text content of element

        Args:
            selector: CSS selector

        Returns:
            Optional[str]: Text content or None
        """
        if not self.page:
            return None

        try:
            element = self.page.locator(selector).first
            return await element.text_content()
        except Exception:
            return None

    async def take_screenshot(self, path: str) -> bool:
        """Take screenshot of current page

        Args:
            path: File path to save screenshot

        Returns:
            bool: True if successful
        """
        if not self.page:
            return False

        try:
            out_path = self._normalize_screenshot_path(path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            await self.page.screenshot(path=str(out_path))
            logger.info(f"Screenshot saved to {out_path}")
            return True
        except Exception as e:
            logger.error(f"Screenshot failed: {e}")
            return False

    async def new_tab(self) -> Optional[Page]:
        """Create a new browser tab/page

        Returns:
            Optional[Page]: New page object or None
        """
        if not self.browser:
            return None

        try:
            new_page = await self.browser.new_page()
            await new_page.set_viewport_size({"width": 1920, "height": 1080})
            logger.info("New tab created")
            return new_page
        except Exception as e:
            logger.error(f"Failed to create new tab: {e}")
            return None

    async def inspect_page(self) -> Dict[str, Any]:
        """Inspect current page state and detect potential issues

        Returns:
            Dict[str, Any]: Inspection results including errors, captcha, etc.
        """
        result = {
            "has_errors": False,
            "has_captcha": False,
            "error_messages": [],
            "page_title": "",
            "url": ""
        }

        if not self.page:
            return result

        try:
            result["page_title"] = await self.page.title()
            result["url"] = self.page.url
            content = await self.page.inner_text('body')
            page_text = content.lower()

            # Check for common error indicators
            error_keywords = [
                '404', '500', 'server error',
                'connection failed', 'timeout', 'blocked', 'forbidden', 'access denied',
                'rate limit', 'too many requests', 'unavailable', 'maintenance'
            ]

            for keyword in error_keywords:
                if keyword in page_text:
                    result["has_errors"] = True
                    result["error_messages"].append(keyword)

            # Check for captcha
            result["has_captcha"] = await self.detect_captcha()

            logger.info(f"Page inspection: errors={result['has_errors']}, captcha={result['has_captcha']}")

        except Exception as e:
            logger.error(f"Page inspection failed: {e}")

        return result

    async def results_visible(self, kind: str = "auto") -> bool:
        if not self.page:
            return False

        try:
            if kind in ("booking", "booking.com", "hotel", "auto"):
                try:
                    cards = await self.page.locator('[data-testid="property-card"], [data-testid="property-card-container"], .sr_property_block').count()
                    if cards and cards > 0:
                        return True
                except Exception:
                    pass

            if kind in ("search", "bing", "auto"):
                try:
                    items = await self.page.locator('.b_algo').count()
                    if items and items > 0:
                        return True
                except Exception:
                    pass

            body = await self.page.inner_text('body')
            body_lower = body.lower() if body else ""
            if kind in ("booking", "booking.com", "hotel", "auto") and any(s in body_lower for s in ["properties found", "exact matches found"]):
                return True
            if kind in ("search", "bing", "auto") and any(s in body_lower for s in ["search results", "results for"]):
                return True

            return False
        except Exception:
            return False

    async def detect_captcha(self) -> bool:
        """Detect if a captcha is present on the page

        Returns:
            bool: True if captcha detected
        """
        if not self.page:
            return False

        try:
            # Check for common captcha selectors
            captcha_selectors = [
                '[class*="captcha"]',
                '[id*="captcha"]',
                '.recaptcha',
                '#recaptcha',
                '[data-sitekey]',  # reCAPTCHA
                '.h-captcha',
                '.hcaptcha',
                '.cloudflare-challenge',
                '.cf-challenge',
                '.cf-browser-verification',
                '.challenge-running',
                '[class*="turnstile"]',  # Cloudflare Turnstile
                '.arkose',  # Arkose Labs
                '.imperva',  # Imperva
                '.incapsula',
                '.ddos-protection',
                '.anti-bot'
            ]

            for selector in captcha_selectors:
                try:
                    element = await self.page.query_selector(selector)
                    if element:
                        logger.info(f"Captcha detected with selector: {selector}")
                        # Take screenshot when captcha detected
                        await self.take_screenshot(f"screenshot_captcha_detected_{int(time.time())}.png")
                        return True
                except:
                    continue

            # Check for text content indicating captcha
            page_text = await self.page.inner_text('body')
            page_lower = page_text.lower() if page_text else ""
            # Avoid overly generic keywords that trigger false positives on normal pages.
            strong_keywords = [
                'captcha',
                'g-recaptcha',
                'hcaptcha',
                'turnstile',
                'prove you are not a robot',
                'verify you are human',
                'cf-challenge',
                'cloudflare challenge',
                'security check',
                'browser verification',
            ]
            if any(keyword in page_lower for keyword in strong_keywords):
                logger.info("Captcha detected via text content")
                return True

            return False

        except Exception as e:
            logger.error(f"Captcha detection failed: {e}")
            return False

    async def solve_recaptcha_v2(self, site_key: str, url: str) -> Optional[str]:
        """Solve reCAPTCHA v2 using 2Captcha

        Args:
            site_key: reCAPTCHA site key
            url: Current page URL

        Returns:
            Optional[str]: Solved token or None
        """
        if not self.captcha_api_key:
            logger.error("No captcha API key provided")
            return None

        try:
            # Submit captcha to 2Captcha
            submit_url = "http://2captcha.com/in.php"
            params = {
                'key': self.captcha_api_key,
                'method': 'userrecaptcha',
                'googlekey': site_key,
                'pageurl': url,
                'json': 1
            }

            response = requests.post(submit_url, data=params, timeout=10)
            result = response.json()

            if result.get('status') != 1:
                logger.error(f"Failed to submit captcha: {result}")
                return None

            request_id = result.get('request')

            # Poll for result
            for _ in range(30):  # Max 30 polls, ~5 minutes
                time.sleep(10)
                res_url = "http://2captcha.com/res.php"
                res_params = {
                    'key': self.captcha_api_key,
                    'action': 'get',
                    'id': request_id,
                    'json': 1
                }

                res_response = requests.get(res_url, params=res_params, timeout=10)
                res_result = res_response.json()

                if res_result.get('status') == 1:
                    token = res_result.get('request')
                    logger.info("Captcha solved successfully")
                    return token
                elif res_result.get('request') == 'CAPCHA_NOT_READY':
                    continue
                else:
                    logger.error(f"Captcha solving failed: {res_result}")
                    return None

            logger.error("Captcha solving timed out")
            return None

        except Exception as e:
            logger.error(f"Captcha solving error: {e}")
            return None

    async def solve_hcaptcha(self, site_key: str, url: str) -> Optional[str]:
        """Solve hCaptcha using 2Captcha

        Args:
            site_key: hCaptcha site key
            url: Current page URL

        Returns:
            Optional[str]: Solved token or None
        """
        if not self.captcha_api_key:
            logger.error("No captcha API key provided")
            return None

        try:
            # Submit captcha to 2Captcha
            submit_url = "http://2captcha.com/in.php"
            params = {
                'key': self.captcha_api_key,
                'method': 'hcaptcha',
                'sitekey': site_key,
                'pageurl': url,
                'json': 1
            }

            response = requests.post(submit_url, data=params, timeout=10)
            result = response.json()

            if result.get('status') != 1:
                logger.error(f"Failed to submit hCaptcha: {result}")
                return None

            request_id = result.get('request')

            # Poll for result
            for _ in range(30):  # Max 30 polls, ~5 minutes
                time.sleep(10)
                res_url = "http://2captcha.com/res.php"
                res_params = {
                    'key': self.captcha_api_key,
                    'action': 'get',
                    'id': request_id,
                    'json': 1
                }

                res_response = requests.get(res_url, params=res_params, timeout=10)
                res_result = res_response.json()

                if res_result.get('status') == 1:
                    token = res_result.get('request')
                    logger.info("hCaptcha solved successfully")
                    return token
                elif res_result.get('request') == 'CAPCHA_NOT_READY':
                    continue
                else:
                    logger.error(f"hCaptcha solving failed: {res_result}")
                    return None

            logger.error("hCaptcha solving timed out")
            return None

        except Exception as e:
            logger.error(f"hCaptcha solving error: {e}")
            return None

    async def handle_captcha(self) -> bool:
        """Attempt to handle any captcha on the current page

        Returns:
            bool: True if captcha handled successfully
        """
        if not await self.detect_captcha():
            return True  # No captcha, success

        if not self.captcha_api_key:
            logger.warning("Captcha detected but no API key provided. Assuming manual handling.")
            return True

        try:
            # Try to find reCAPTCHA
            recaptcha_element = await self.page.query_selector('[data-sitekey]')
            if recaptcha_element:
                site_key = await recaptcha_element.get_attribute('data-sitekey')
                current_url = self.page.url
                token = await self.solve_recaptcha_v2(site_key, current_url)

                if token:
                    # Inject the token into the page
                    await self.page.evaluate(f"""
                        document.querySelector('[data-sitekey]').setAttribute('data-callback', 'onCaptchaSolved');
                        window.onCaptchaSolved = function(token) {{}};
                        grecaptcha.execute('{site_key}', {{action: 'submit'}}).then(function(token) {{
                            window.captchaToken = token;
                        }});
                    """)
                    # Wait a bit and set the token
                    await asyncio.sleep(2)
                    await self.page.evaluate(f"window.captchaToken = '{token}';")
                    logger.info("reCAPTCHA token injected")
                    # Take screenshot after captcha handling
                    await self.take_screenshot(f"screenshot_captcha_handled_{int(time.time())}.png")
                    return True

            # Try to find hCaptcha
            hcaptcha_element = await self.page.query_selector('.h-captcha, .hcaptcha')
            if hcaptcha_element:
                site_key = await hcaptcha_element.get_attribute('data-sitekey')
                current_url = self.page.url
                token = await self.solve_hcaptcha(site_key, current_url)

                if token:
                    # Inject the token into the hCaptcha response field
                    await self.page.evaluate(f"""
                        const hcaptchaInput = document.querySelector('.h-captcha input[type="hidden"], .hcaptcha input[type="hidden"]');
                        if (hcaptchaInput) {{
                            hcaptchaInput.value = '{token}';
                        }}
                    """)
                    logger.info("hCaptcha token injected")
                    return True

            # For other captchas, take screenshot and try to solve (placeholder)
            screenshot_path = f"/tmp/captcha_{int(time.time())}.png"
            await self.take_screenshot(screenshot_path)
            logger.info(f"Captcha screenshot saved to {screenshot_path}")
            # Here you could integrate image-based captcha solving
            return False  # Placeholder, assume failure for non-reCAPTCHA

        except Exception as e:
            logger.error(f"Captcha handling failed: {e}")
            return False

    async def search_google(self, query: str) -> List[Dict[str, Any]]:
        """Search Google and extract results

        Args:
            query: Search query

        Returns:
            List[Dict[str, Any]]: Search results with title, url, snippet
        """
        results = []

        if not self.page:
            return results

        try:
            # Navigate to DuckDuckGo (more automation-friendly than Google)
            await self.page.goto("https://duckduckgo.com/", wait_until="networkidle")
            # Take screenshot after navigation to DuckDuckGo
            await self.take_screenshot(f"screenshot_duckduckgo_navigate_{int(time.time())}.png")
            # Inspect page for errors
            inspection = await self.inspect_page()
            if inspection["has_errors"]:
                logger.warning(f"Errors detected on DuckDuckGo page: {inspection['error_messages']}")

            if await self.detect_captcha():
                if not await self.handle_captcha():
                    logger.warning("Captcha detected and failed to handle")
                    return results

            # Directly navigate to Bing search results
            await self.page.goto(f"https://www.bing.com/search?q={query}", wait_until="domcontentloaded", timeout=30000)
            # Take screenshot after search
            await self.take_screenshot(f"screenshot_bing_search_{int(time.time())}.png")
            # Inspect page for errors
            inspection = await self.inspect_page()
            if inspection["has_errors"]:
                logger.warning(f"Errors detected on search results page: {inspection['error_messages']}")

            # Extract organic results using Bing selectors
            result_selectors = [
                '.b_algo',  # Bing result containers
                'h2 a',  # Fallback to title links
            ]

            results = []
            for selector in result_selectors:
                try:
                    containers = await self.page.locator(selector).all()
                    for container in containers[:10]:
                        try:
                            if selector == 'h2 a':
                                title = await container.text_content()
                                url = await container.get_attribute('href')
                            else:
                                title_elem = container.locator('h2 a').first
                                title = await title_elem.text_content() if await title_elem.count() > 0 else ''
                                url_elem = container.locator('h2 a').first
                                url = await url_elem.get_attribute('href') if await url_elem.count() > 0 else ''

                            title = title.strip() if title else ''
                            if title and url and len(title) > 3 and not url.startswith('javascript:'):
                                results.append({
                                    'title': title,
                                    'url': url,
                                    'snippet': '',
                                    'source': 'duckduckgo'
                                })
                        except:
                            continue

                    if results:
                        break  # Found results with this selector

                except Exception as e:
                    logger.warning(f"Selector {selector} failed: {e}")
                    continue

            logger.info(f"Extracted {len(results)} DuckDuckGo search results for query: {query}")
            # Take screenshot after results extraction
            await self.take_screenshot(f"screenshot_duckduckgo_results_{int(time.time())}.png")

        except Exception as e:
            logger.error(f"DuckDuckGo search failed: {e}")

        return results

    async def multi_tab_search(self, queries: List[str]) -> List[Dict[str, Any]]:
        """Perform Google searches in multiple tabs simultaneously

        Args:
            queries: List of search queries

        Returns:
            List[Dict[str, Any]]: Aggregated search results from all queries
        """
        if not self.browser:
            return []

        async def search_in_tab(query: str) -> List[Dict[str, Any]]:
            """Search in a single tab"""
            new_page = await self.browser.new_page()
            await new_page.set_viewport_size({"width": 1920, "height": 1080})
            try:
                # Navigate to Google
                await new_page.goto("https://www.google.com", wait_until="networkidle")
                # Take screenshot
                await new_page.screenshot(path=f"screenshot_multi_tab_navigate_{query.replace(' ', '_')}_{int(time.time())}.png")

                # Check for captcha
                captcha_detected = await self.detect_captcha_on_page(new_page)
                if captcha_detected:
                    if not await self.handle_captcha_on_page(new_page):
                        logger.warning(f"CAPTCHA failed in tab for query: {query}")
                        return []

                # Search
                search_box = new_page.locator('textarea[name="q"], input[name="q"]').first
                await search_box.fill(query)
                await search_box.press('Enter')
                await asyncio.sleep(10)
                # Take screenshot
                await new_page.screenshot(path=f"screenshot_multi_tab_search_{query.replace(' ', '_')}_{int(time.time())}.png")

                # Extract results
                h3_locators = await new_page.locator('h3').all()
                results = []
                for locator in h3_locators[:5]:  # Limit per query
                    title = await locator.text_content()
                    title = title.strip()
                    link_locator = locator.locator('xpath=ancestor::a').first
                    url = await link_locator.get_attribute('href') if await link_locator.count() > 0 else ''
                    if title and url:
                        results.append({
                            'title': title,
                            'url': url,
                            'snippet': '',
                            'source': 'google',
                            'query': query
                        })

                # Take screenshot after results
                await new_page.screenshot(path=f"screenshot_multi_tab_results_{query.replace(' ', '_')}_{int(time.time())}.png")
                return results

            finally:
                await new_page.close()

        # Run searches in parallel
        tasks = [search_in_tab(query) for query in queries]
        all_results = await asyncio.gather(*tasks)

        # Flatten results
        flattened_results = []
        for result_list in all_results:
            flattened_results.extend(result_list)

        logger.info(f"Multi-tab search completed for {len(queries)} queries, total results: {len(flattened_results)}")
        return flattened_results

    async def detect_captcha_on_page(self, page) -> bool:
        """Detect captcha on a specific page"""
        try:
            captcha_selectors = [
                '[class*="captcha"]', '[id*="captcha"]', '.recaptcha', '#recaptcha',
                '[data-sitekey]', '.h-captcha', '.hcaptcha', '.cloudflare-challenge',
                '.cf-challenge', '.cf-browser-verification', '.challenge-running',
                '[class*="turnstile"]', '.arkose', '.imperva', '.incapsula',
                '.ddos-protection', '.anti-bot'
            ]
            for selector in captcha_selectors:
                element = await page.query_selector(selector)
                if element:
                    return True
            page_text = await page.inner_text('body')
            captcha_keywords = ['captcha', 'verification', 'robot', 'human', 'prove you are not a robot', 'cloudflare', 'challenge', 'security check', 'anti-bot', 'ddos protection', 'verification code', 'please wait']
            if any(keyword in page_text.lower() for keyword in captcha_keywords):
                return True
            return False
        except Exception:
            return False

    async def handle_captcha_on_page(self, page) -> bool:
        """Handle captcha on a specific page (simplified)"""
        # For simplicity, just take screenshot and return False
        await page.screenshot(path=f"screenshot_captcha_multi_tab_{int(time.time())}.png")
        return False

    async def crawl_dfs(self, start_url: str, max_depth: int = 2, max_pages: int = 10) -> List[Dict[str, Any]]:
        """Depth-first search web crawler

        Args:
            start_url: Starting URL to start crawling
            max_depth: Maximum depth to crawl
            max_pages: Maximum number of pages to crawl

        Returns:
            List[Dict[str, Any]]: List of crawled page data
        """
        visited = set()
        stack = [(start_url, 0)]  # (url, depth)
        crawled_pages = []

        while stack and len(crawled_pages) < max_pages:
            url, depth = stack.pop()

            if url in visited or depth > max_depth:
                continue

            visited.add(url)

            logger.info(f"Navigating to {url} at depth {depth}")

            if not await self.navigate(url):
                logger.warning(f"Failed to navigate to {url}")
                continue

            # Take screenshot after navigation
            await self.take_screenshot(f"screenshot_crawl_navigate_{int(time.time())}.png")
            # Inspect page for errors
            inspection = await self.inspect_page()
            if inspection["has_errors"]:
                logger.warning(f"Errors detected on crawled page {url}: {inspection['error_messages']}")

            # Handle CAPTCHA if detected
            if await self.detect_captcha():
                if not await self.handle_captcha():
                    logger.warning(f"CAPTCHA handling failed for {url}")
                    continue

            # Extract page information
            title = await self.page.title()
            content = await self.page.inner_text('body')

            # Extract links
            links = await self.page.query_selector_all('a[href]')
            link_urls = []
            for link in links[:50]:  # Limit to prevent excessive links
                href = await link.get_attribute('href')
                if href:
                    if not href.startswith('http'):
                        href = urllib.parse.urljoin(url, href)
                    if href.startswith('http') and href not in visited:
                        link_urls.append(href)

            crawled_pages.append({
                'url': url,
                'title': title,
                'depth': depth,
                'content': content[:2000],  # Truncate content for summary
                'links': link_urls[:20]  # Limit stored links
            })

            logger.info(f"Crawled {url}, found {len(link_urls)} links")

            # Add new links to stack for DFS (reverse to maintain order)
            for link in reversed(link_urls):
                if link not in visited:
                    stack.append((link, depth + 1))

        logger.info(f"Crawling completed, crawled {len(crawled_pages)} pages")
        return crawled_pages

    async def execute_booking_workflow(self, booking_details: Dict[str, Any]) -> Dict[str, Any]:
        """Execute autonomous booking workflow

        Args:
            booking_details: Booking parameters

        Returns:
            Dict[str, Any]: Workflow results
        """
        results = {"success": False, "step": "initiated"}

        try:
            # Navigate to booking site
            site_url = booking_details.get("site_url")
            if not site_url or not await self.navigate(site_url):
                return {"success": False, "error": "Navigation failed"}

            results["step"] = "navigated"
            # Take screenshot after navigation
            await self.take_screenshot(f"screenshot_booking_navigate_{int(time.time())}.png")
            # Inspect page
            inspection = await self.inspect_page()
            if inspection["has_errors"]:
                logger.warning(f"Errors detected on booking site: {inspection['error_messages']}")

            # Handle any captchas before proceeding
            if not await self.handle_captcha():
                return {"success": False, "error": "Captcha handling failed"}

            results["step"] = "captcha_handled"
            search_fields = booking_details.get("search_fields", {})
            if not await self.fill_form(search_fields):
                return {"success": False, "error": "Search form fill failed"}

            results["step"] = "search_filled"
            # Take screenshot after search fill
            await self.take_screenshot(f"screenshot_booking_search_fill_{int(time.time())}.png")

            # Click search button
            search_button = booking_details.get("search_button")
            if search_button and not await self.search_and_click(search_button):
                return {"success": False, "error": "Search button click failed"}

            results["step"] = "search_clicked"
            # Take screenshot after search click
            await self.take_screenshot(f"screenshot_booking_search_click_{int(time.time())}.png")

            # Wait for results and select option
            result_selector = booking_details.get("result_selector")
            if result_selector and await self.wait_for_element(result_selector):
                await self.search_and_click(result_selector)
                results["step"] = "option_selected"
                # Take screenshot after option select
                await self.take_screenshot(f"screenshot_booking_option_select_{int(time.time())}.png")

            # Fill booking details
            booking_fields = booking_details.get("booking_fields", {})
            if booking_fields and await self.fill_form(booking_fields):
                results["step"] = "booking_filled"
                # Take screenshot after booking fill
                await self.take_screenshot(f"screenshot_booking_fill_{int(time.time())}.png")

            # Stop at payment confirmation
            results["success"] = True
            results["message"] = "Ready for payment confirmation"

        except Exception as e:
            results["error"] = str(e)

        return results
