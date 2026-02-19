#!/usr/bin/env python3
"""
Test script for J.A.S.O.N. Browser Agent
"""

import asyncio
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from jason.tools.browser_agent import BrowserAgent

async def test_browser():
    """Test browser automation functionality"""
    print("Testing J.A.S.O.N. Browser Agent...")

    async with BrowserAgent(headless=False) as agent:
        print("✓ Browser agent initialized")

        # Test navigation to Google
        print("Navigating to Google...")
        success = await agent.navigate("https://www.google.com")
        if success:
            print("✓ Successfully navigated to Google")
        else:
            print("✗ Failed to navigate to Google")
            return

        # Test search
        print("Performing search...")
        results = await agent.search_google("test query")
        if results:
            print(f"✓ Found {len(results)} search results")
            for i, result in enumerate(results[:3]):
                print(f"  {i+1}. {result.get('title', 'No title')}")
        else:
            print("✗ Search failed")

        # Take screenshot
        screenshot_path = "test_screenshot.png"
        success = await agent.take_screenshot(screenshot_path)
        if success:
            print(f"✓ Screenshot saved to {screenshot_path}")
        else:
            print("✗ Screenshot failed")

    print("Browser test completed")

if __name__ == "__main__":
    asyncio.run(test_browser())
