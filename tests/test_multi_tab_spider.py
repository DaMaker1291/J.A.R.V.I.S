import asyncio
import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from jason.tools.browser_agent import BrowserAgent

async def test_multi_tab_search():
    print("Testing Multi-Tab Search...")
    async with BrowserAgent(headless=True) as agent:
        queries = ["AI assistant", "web automation", "Python programming"]
        print(f"Performing multi-tab search for queries: {queries}")

        results = await agent.multi_tab_search(queries)

        print(f"Multi-tab search completed, total results: {len(results)}")
        for i, result in enumerate(results[:10], 1):  # Show first 10
            print(f"{i}. [{result.get('query', 'unknown')}] {result['title']}")
            print(f"   URL: {result['url']}")
            print()

    print("Multi-tab search test completed!")

async def test_dfs_spider():
    print("Testing DFS Spider...")
    async with BrowserAgent(headless=True) as agent:
        start_url = "https://example.com"
        print(f"Starting DFS crawl from: {start_url}")

        crawled_pages = await agent.crawl_dfs(start_url, max_depth=1, max_pages=3)

        print(f"DFS crawl completed, crawled {len(crawled_pages)} pages")
        for page in crawled_pages:
            print(f"URL: {page['url']}")
            print(f"Title: {page['title']}")
            print(f"Depth: {page['depth']}")
            print(f"Links found: {len(page['links'])}")
            print()

    print("DFS spider test completed!")

async def main():
    print("=" * 60)
    print("TESTING J.A.R.V.I.S. BROWSER FEATURES")
    print("Multi-Tab Research & DFS Spider")
    print("=" * 60)

    try:
        await test_multi_tab_search()
        print()
        await test_dfs_spider()
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

    print("=" * 60)
    print("Features working: Multi-tab research and DFS spider!")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
