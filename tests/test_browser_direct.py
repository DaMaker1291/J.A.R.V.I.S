import asyncio
import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from jason.tools.browser_agent import BrowserAgent

async def test_browser_directly():
    print("Testing BrowserAgent directly...")
    try:
        async with BrowserAgent(headless=False) as agent:
            print("BrowserAgent initialized successfully")
            success = await agent.navigate("https://www.google.com")
            if success:
                print("Successfully navigated to Google")
                await asyncio.sleep(2)
                screenshot_success = await agent.take_screenshot("direct_test_screenshot.png")
                if screenshot_success:
                    print("Screenshot saved to direct_test_screenshot.png")
                else:
                    print("Screenshot failed")
            else:
                print("Navigation failed")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_browser_directly())
