"""
J.A.S.O.N. Browser Agent Tool
Concierge Module - Autonomous Web Interaction
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

from playwright.async_api import async_playwright, Browser, Page, Playwright

logger = logging.getLogger(__name__)

class BrowserAgent:
    """Autonomous browser agent using Playwright"""

    def __init__(self, headless: bool = True):
        self.headless = headless
        self.playwright: Optional[Playwright] = None
        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None

    async def __aenter__(self):
        """Async context manager entry"""
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=self.headless)
        self.page = await self.browser.new_page()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.page:
            await self.page.close()
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()

    async def navigate(self, url: str) -> bool:
        """Navigate to URL

        Args:
            url: Target URL

        Returns:
            bool: True if navigation successful
        """
        if not self.page:
            return False

        try:
            await self.page.goto(url, wait_until="networkidle")
            logger.info(f"Navigated to {url}")
            return True
        except Exception as e:
            logger.error(f"Navigation failed: {e}")
            return False

    async def search_and_click(self, selector: str, text: Optional[str] = None) -> bool:
        """Search for element and click it

        Args:
            selector: CSS selector
            text: Optional text to match

        Returns:
            bool: True if element found and clicked
        """
        if not self.page:
            return False

        try:
            if text:
                element = self.page.locator(selector).filter(has_text=text).first
            else:
                element = self.page.locator(selector).first

            await element.click()
            logger.info(f"Clicked element: {selector}")
            return True
        except Exception as e:
            logger.error(f"Click failed: {e}")
            return False

    async def fill_form(self, fields: Dict[str, str]) -> bool:
        """Fill form fields

        Args:
            fields: Dict of selector -> value

        Returns:
            bool: True if all fields filled
        """
        if not self.page:
            return False

        try:
            for selector, value in fields.items():
                await self.page.fill(selector, value)
                logger.info(f"Filled {selector} with {value}")
            return True
        except Exception as e:
            logger.error(f"Form fill failed: {e}")
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
            await self.page.screenshot(path=path)
            logger.info(f"Screenshot saved to {path}")
            return True
        except Exception as e:
            logger.error(f"Screenshot failed: {e}")
            return False

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

            # Fill search form
            search_fields = booking_details.get("search_fields", {})
            if not await self.fill_form(search_fields):
                return {"success": False, "error": "Search form fill failed"}

            results["step"] = "search_filled"

            # Click search button
            search_button = booking_details.get("search_button")
            if search_button and not await self.search_and_click(search_button):
                return {"success": False, "error": "Search button click failed"}

            results["step"] = "search_clicked"

            # Wait for results and select option
            result_selector = booking_details.get("result_selector")
            if result_selector and await self.wait_for_element(result_selector):
                await self.search_and_click(result_selector)
                results["step"] = "option_selected"

            # Fill booking details
            booking_fields = booking_details.get("booking_fields", {})
            if booking_fields and await self.fill_form(booking_fields):
                results["step"] = "booking_filled"

            # Stop at payment confirmation
            results["success"] = True
            results["message"] = "Ready for payment confirmation"

        except Exception as e:
            results["error"] = str(e)

        return results
