import asyncio
import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from jason.tools.browser_agent import BrowserAgent

async def test_dfs_spider():
    print("Testing DFS Spider...")
    async with BrowserAgent(headless=True) as agent:
        start_url = "https://httpbin.org/htm"
        print(f"Starting DFS crawl from: {start_url}")

        crawled_pages = await agent.crawl_dfs(start_url, max_depth=2, max_pages=5)

        print(f"DFS crawl completed, crawled {len(crawled_pages)} pages")
        for page in crawled_pages:
            print(f"URL: {page['url']}")
            print(f"Title: {page['title']}")
            print(f"Depth: {page['depth']}")
            print(f"Links found: {len(page['links'])}")
            print()

    print("DFS spider test completed!")

if __name__ == "__main__":
    asyncio.run(test_dfs_spider())
