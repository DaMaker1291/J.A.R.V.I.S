import asyncio
import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from jason.tools.browser_agent import BrowserAgent

async def test_browser():
    print("Testing BrowserAgent...")
    async with BrowserAgent(headless=True) as agent:
        success = await agent.navigate("https://www.google.com")
        if success:
            print("Successfully navigated to Google")
            await asyncio.sleep(2)
            await agent.take_screenshot("test_browser.png")
            print("Screenshot saved to test_browser.png")

            # Perform search to test more functionality
            results = await agent.search_google("AI assitant")
            print(f"Search completed, found {len(results)} results")

            # Take final screenshot
            await agent.take_screenshot("test_browser_after_search.png")
            print("Final screenshot saved to test_browser_after_search.png")
        else:
            print("Failed to navigate to Google")

if __name__ == "__main__":
    asyncio.run(test_browser())
