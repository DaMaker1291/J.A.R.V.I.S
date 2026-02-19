import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from jason.core.swarm import SwarmManager

def verify_real_actions():
    print("Verifying real actions in SwarmManager...")
    
    # Initialize SwarmManager (config-less for mock test, but logic should still trigger)
    config = {
        'api_keys': {'gemini': ''},
        'zero_api_mode': True
    }
    swarm = SwarmManager(config=config)
    
    commands_to_test = [
        "renew my OCI",
        "book a 5 star cheap holiday to Japan",
        "apply for a passport"
    ]
    
    for cmd in commands_to_test:
        print(f"\nTesting Command: '{cmd}'")
        result = swarm.process_command(cmd)
        print(f"Result: {result}")
        
        # Verify result contains 'REAL ACTION INITIATED' or similar
        if "REAL ACTION" in result or "Initiating real" in result:
            print(f"✓ SUCCESS: Real action triggered for '{cmd}'")
        else:
            print(f"✗ FAILURE: Fake or incorrect response for '{cmd}'")

if __name__ == "__main__":
    verify_real_actions()
