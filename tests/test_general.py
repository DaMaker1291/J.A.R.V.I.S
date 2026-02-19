"""
Test script for J.A.S.O.N. General Assistant
Tests various request types to ensure the app can handle any request
"""

import asyncio
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from jason.modules.general_assistant import GeneralAssistant

def test_general_assistant():
    """Test the general assistant with various request types"""

    # Dummy config - in real use, load from config.yaml
    config = {
        'gemini_api_key': os.environ.get('GEMINI_API_KEY', 'dummy_key'),  # Will fail without real key
        'captcha_api_key': 'dummy_captcha_key',
        'vpn': {'provider': 'nordvpn'},
        'arbitrage': {'countries': ['us', 'uk', 'jp']},
        'deals': {'sites': ['https://www.expedia.com', 'https://www.groupon.com']},
        'desktop': {'safety_bounds': {'min_x': 100, 'max_x': 1800, 'min_y': 100, 'max_y': 1000}}
    }

    assistant = GeneralAssistant(config)

    test_requests = [
        "Book a holiday to Japan for 7 days",
        "Do my homework - open Teams and check for assignments",
        "Type this essay human-like: 'The impact of AI on modern society is profound...'",
        "Open Microsoft Teams application",
        "Automate my dropshipping business - check orders and update inventory",
        "Take a screenshot of the current screen",
        "Move mouse to position 500, 500",
        "Search for cheap flights from New York to Tokyo"
    ]

    print("Testing J.A.S.O.N. General Assistant...")
    print("=" * 50)

    for i, request in enumerate(test_requests, 1):
        print(f"\nTest {i}: {request}")
        print("-" * 30)

        try:
            result = assistant.execute_request(request)
            print(f"Success: {result.get('success', False)}")
            if result.get('needs_clarification'):
                print(f"Question: {result.get('question')}")
            else:
                print(f"Response: {result.get('response', 'No response')[:200]}...")
                print(f"Steps executed: {result.get('executed_steps', 0)}")
                print(f"Tools used: {result.get('tools_used', [])}")

        except Exception as e:
            print(f"Error: {e}")

    print("\n" + "=" * 50)
    print("Testing completed!")

if __name__ == "__main__":
    test_general_assistant()
