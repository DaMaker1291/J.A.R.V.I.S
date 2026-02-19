#!/usr/bin/env python3
"""
Test script for J.A.S.O.N. SwarmManager command processing
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from jason.core.swarm import SwarmManager

def test_command_processing():
    """Test command processing functionality"""
    print("Testing J.A.S.O.N. SwarmManager command processing...")

    # Initialize SwarmManager
    swarm = SwarmManager()
    print("✓ SwarmManager initialized")

    # Test various commands
    test_commands = [
        "search for python tutorials",
        "navigate to google.com",
        "find information about AI",
        "book a flight to japan"
    ]

    for command in test_commands:
        print(f"\nTesting command: '{command}'")
        try:
            result = swarm.process_command(command)
            print(f"Result: {result[:200]}...")
            if "failed" in result.lower() or "error" in result.lower():
                print("❌ Command failed")
            else:
                print("✅ Command processed")
        except Exception as e:
            print(f"❌ Exception: {e}")

    print("\nCommand processing test completed")

if __name__ == "__main__":
    test_command_processing()
