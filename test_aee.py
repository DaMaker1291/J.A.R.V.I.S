import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from jason.core.swarm import SwarmManager
import json

def test_commands():
    # Force some environment variables for better testing if needed
    os.environ["PYAUTOGUI_GHOST_MODE"] = "1" 
    
    manager = SwarmManager()
    
    commands = [
        "Book a 5-day holiday to Tokyo for two people in June",
        "Start a dropshipping business for eco-friendly yoga mats",
        "Make a CAD prototype of a lamp",
        "Check system status and performance",
        "Research the latest AI trends in 2026",
        "Scrape products from https://apple.com/iphone",
        "Fill out the worksheet structure of medical_form.pdf",
        "What is my current disk usage?"
    ]
    
    for cmd in commands:
        print(f"\n[TESTING COMMAND]: {cmd}")
        try:
            response = manager.process_command(cmd)
            print(f"[RESPONSE]:\n{response}")
            print("-" * 50)
        except Exception as e:
            import traceback
            print(f"[ERROR]: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    test_commands()
