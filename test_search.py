#!/usr/bin/env python3
"""Test the updated search functionality"""

from jason.tools.serp_api import SearXNGSearch

def test_search():
    search_tool = SearXNGSearch()
    result = search_tool.search("test query")
    print("Search result:", result)
    return result

if __name__ == "__main__":
    test_search()
