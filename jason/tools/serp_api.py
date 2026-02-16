"""
J.A.S.O.N. SearXNG Tool
Concierge Module - Local Sovereign Web Search
"""

import requests
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class SearXNGSearch:
    """Local search tool using SearXNG"""

    def __init__(self, base_url: str = "http://localhost:8080"):
        """Initialize with SearXNG base URL

        Args:
            base_url: SearXNG server URL
        """
        self.base_url = base_url.rstrip('/')

    def search(self, query: str, **kwargs) -> Dict[str, Any]:
        """Perform web search using local SearXNG

        Args:
            query: Search query
            **kwargs: Additional search parameters

        Returns:
            Dict[str, Any]: Search results
        """
        params = {
            "q": query,
            "format": "json",
            **kwargs
        }

        try:
            response = requests.get(f"{self.base_url}/search", params=params, timeout=30)
            response.raise_for_status()

            data = response.json()
            logger.info(f"SearXNG search completed for query: {query}")
            return data

        except requests.RequestException as e:
            logger.error(f"SearXNG search failed: {e}")
            return {"error": str(e)}

    def search_flights(self, origin: str, destination: str, date: str, **kwargs) -> Dict[str, Any]:
        """Search for flights using SearXNG

        Args:
            origin: Origin airport code
            destination: Destination airport code
            date: Departure date (YYYY-MM-DD)
            **kwargs: Additional parameters

        Returns:
            Dict[str, Any]: Flight search results
        """
        query = f"flights from {origin} to {destination} on {date}"
        return self.search(query, **kwargs)

    def search_hotels(self, location: str, checkin: str, checkout: str, **kwargs) -> Dict[str, Any]:
        """Search for hotels using SearXNG

        Args:
            location: Hotel location
            checkin: Check-in date (YYYY-MM-DD)
            checkout: Check-out date (YYYY-MM-DD)
            **kwargs: Additional parameters

        Returns:
            Dict[str, Any]: Hotel search results
        """
        query = f"hotels in {location} from {checkin} to {checkout}"
        return self.search(query, **kwargs)

    def extract_flight_info(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract flight information from SearXNG results

        Args:
            results: SearXNG results

        Returns:
            List[Dict[str, Any]]: List of flight options
        """
        flights = []

        try:
            # SearXNG returns results in 'results' array
            search_results = results.get("results", [])
            for result in search_results[:5]:  # Top 5 results
                # Extract relevant info from title/content
                title = result.get("title", "")
                content = result.get("content", "")
                url = result.get("url", "")

                # Parse for flight info (simplified)
                if "flight" in title.lower() or "flight" in content.lower():
                    flights.append({
                        "airline": "Various",  # Would need better parsing
                        "flight_number": "N/A",
                        "departure": "N/A",
                        "arrival": "N/A",
                        "duration": "N/A",
                        "price": "Check site",
                        "departure_time": "N/A",
                        "arrival_time": "N/A",
                        "url": url,
                        "title": title
                    })
        except Exception as e:
            logger.error(f"Failed to extract flight info from SearXNG: {e}")

        return flights

    def extract_hotel_info(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract hotel information from SearXNG results

        Args:
            results: SearXNG results

        Returns:
            List[Dict[str, Any]]: List of hotel options
        """
        hotels = []

        try:
            search_results = results.get("results", [])
            for result in search_results[:5]:  # Top 5 results
                title = result.get("title", "")
                content = result.get("content", "")
                url = result.get("url", "")

                if "hotel" in title.lower() or "hotel" in content.lower():
                    hotels.append({
                        "name": title.split(" - ")[0] if " - " in title else title,
                        "rating": "N/A",
                        "price": "Check site",
                        "address": "N/A",
                        "description": content[:200] + "..." if len(content) > 200 else content,
                        "url": url
                    })
        except Exception as e:
            logger.error(f"Failed to extract hotel info from SearXNG: {e}")

        return hotels
