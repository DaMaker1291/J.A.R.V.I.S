#!/usr/bin/env python3
"""
Test script for J.A.S.O.N. booking functionality
"""

import asyncio
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from jason.core.swarm import SwarmManager

def test_booking():
    """Test booking functionality"""
    print("Testing J.A.S.O.N. booking functionality...")

    # Initialize swarm manager (no API keys for zero-API mode)
    swarm = SwarmManager(config={})

    # Test booking command with group size
    booking_command = "Book a holiday for 5 to Japan next month"
    print(f"Processing command: {booking_command}")

    result = swarm.process_command(booking_command)
    print(f"Result: {result}")

    # Test web search
    web_command = "Search Google for best restaurants in Tokyo"
    print(f"Processing command: {web_command}")

    result2 = swarm.process_command(web_command)
    print(f"Result: {result2}")

    # Test typing task
    typing_command = "Type an essay about artificial intelligence"
    print(f"Processing command: {typing_command}")

    result3 = swarm.process_command(typing_command)
    print(f"Result: {result3}")

if __name__ == "__main__":
    test_booking()
