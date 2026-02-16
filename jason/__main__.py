#!/usr/bin/env python3
"""
J.A.S.O.N. - Just Another Super-Operative Network
A multi-agent autonomous ecosystem inspired by J.A.R.V.I.S.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional

import click
import yaml
from dotenv import load_dotenv

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import J.A.S.O.N. components
from jason.core.swarm import SwarmManager
from jason.core.vision import VisionManager
from jason.core.audio import AudioManager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("jason.log")
    ]
)
logger = logging.getLogger(__name__)

def load_config():
    """Load configuration from config.yaml or create default - supports zero-API mode"""
    config_path = Path('config.yaml')
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        # Create default config with optional API keys for zero-API support
        default_config = {
            'name': 'J.A.S.O.N.',
            'version': '5.0.0',
            # API keys for full functionality (optional - zero-API mode still works)
            'api_keys': {
                'gemini': '',      # For AI vision & general tasks
                'claude': '',      # For advanced reasoning
                'openai': '',      # For enhanced AI capabilities
                'elevenlabs': '',   # For high-quality TTS
                'zoom': '',        # For creating actual Zoom meetings
                'amadeus': '',     # For real flight bookings
                'google_calendar': '', # For Google Calendar integration
                'outlook': '',     # For Outlook calendar integration
                'zapier': '',      # For automation workflows
                'make': ''         # For advanced automation
            },
            # Zero-API configuration (default mode)
            'zero_api_mode': True,  # Enable deterministic processing without APIs
            'searxng_url': 'http://localhost:8080',  # For web search (self-hosted)
            'vpn_providers': ['nordvpn', 'mullvad', 'expressvpn'],  # Auto-detected VPN clients
            'workflow_automation': {
                'enabled': True,
                'travel_booking': True,
                'calendar_scheduling': True,
                'file_management': True,
                'system_maintenance': True
            },
            # Real API integrations (when API keys provided)
            'real_integrations': {
                'zoom_meetings': False,      # Auto-enabled when zoom API key present
                'flight_booking': False,     # Auto-enabled when amadeus API key present  
                'calendar_sync': False,      # Auto-enabled when calendar API keys present
                'automation': False          # Auto-enabled when zapier/make API keys present
            },
            'vision': {
                'enabled': True,
                'model': 'local'  # Use local vision models instead of API
            },
            'audio': {
                'enabled': True,
                'voice_recognition': True,
                'tts_engine': 'piper',  # Zero-API TTS priority
                'military_persona': True
            },
            'security': {
                'firewall_enabled': True,
                'network_monitoring': True,
                'biometric_voice_lock': True
            },
            'protocols': {
                'oracle': {
                    'enabled': True,
                    'default_simulations': 1000,
                    'use_deterministic': True  # Local Monte Carlo without APIs
                },
                'watchtower': {
                    'enabled': True,
                    'monitoring_interval': 300,
                    'use_local_feeds': True  # Local news feeds instead of APIs
                },
                'cipher': {
                    'enabled': True,
                    'voice_analysis': True,
                    'use_local_models': True  # Local voice analysis
                }
            },
            'security': {
                'firewall_enabled': True,
                'network_monitoring': True,
                'biometric_voice_lock': True
            },
            'protocols': {
                'oracle': {
                    'enabled': True,
                    'default_simulations': 1000,
                    'use_deterministic': True  # Local Monte Carlo without APIs
                },
                'watchtower': {
                    'enabled': True,
                    'monitoring_interval': 300,
                    'use_local_feeds': True  # Local news feeds instead of APIs
                },
                'cipher': {
                    'enabled': True,
                    'voice_analysis': True,
                    'use_local_models': True  # Local voice analysis
                }
            }
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)
        
        return default_config

class JASON:
    """Main J.A.S.O.N. class that orchestrates the multi-agent ecosystem."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize J.A.S.O.N. with configuration."""
        self.config = self._load_config(config_path)
        self.swarm = SwarmManager(
            gemini_api_key=self.config.get('api_keys', {}).get('gemini', ''),
            claude_api_key=self.config.get('api_keys', {}).get('claude', ''),
            config=self.config  # Pass full config for zero-API features
        )
        self.vision = VisionManager(self.config.get('api_keys', {}).get('gemini', ''))
        self.audio = AudioManager(self.config.get('api_keys', {}).get('elevenlabs', ''))

        # Start vision monitoring if enabled
        if self.config.get('vision', {}).get('enabled', False):
            self.vision.start_continuous_vision()

        logger.info(f"Initialized {self.config.get('name', 'J.A.S.O.N.')} v{self.config.get('version', '0.1.0')}")

    def _load_config(self, config_path: Optional[str] = None) -> dict:
        """Load configuration from YAML file."""
        try:
            return load_config()
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return {
                'name': 'J.A.S.O.N.',
                'version': '0.1.0',
                'api_keys': {
                    'gemini': '',
                    'claude': '',
                    'elevenlabs': ''
                }
            }

    def start(self):
        """Start the J.A.S.O.N. main loop."""
        logger.info("Starting J.A.S.O.N. (Press Ctrl+C to exit)")
        print(f"\n{'='*50}")
        print(f"  {self.config.get('name', 'J.A.S.O.N.')} v{self.config.get('version', '0.1.0')}")
        print("  Type 'help' for available commands")
        print("  Press Ctrl+C to exit")
        print("="*50 + "\n")

        try:
            while True:
                try:
                    # Voice or text input
                    if self.config.get('voice_input', {}).get('enabled', False):
                        command = self.audio.listen()
                        if command.startswith("Heard:"):
                            command = command[6:].strip()  # Remove "Heard:" prefix
                        else:
                            continue  # No speech detected
                    else:
                        command = input("You: ").strip()

                    if not command:
                        continue

                    # Check biometric voice-lock before processing commands
                    if self.audio.voice_lock_enabled and not self.audio.is_voice_verified():
                        print("J.A.S.O.N.: Biometric Voice-Lock active. Voice verification required.")
                        verification_result = self.audio.listen(timeout=3)
                        if "Voice verification failed" in verification_result:
                            print("J.A.S.O.N.: Voice verification failed. Access denied.")
                            continue
                        else:
                            print("J.A.S.O.N.: Voice verified. Access granted.")

                    if command.lower() in ['exit', 'quit', 'bye']:
                        print("J.A.S.O.N.: Goodbye!")
                        break

                    # Process command through swarm
                    result = self.swarm.process_command(command)

                    # Output result
                    print(f"J.A.S.O.N.: {result}")

                    # Voice output if enabled
                    if self.config.get('voice_output', {}).get('enabled', False):
                        self.audio.speak(result)

                except KeyboardInterrupt:
                    print("\nJ.A.S.O.N.: Shutting down...")
                    break
                except Exception as e:
                    logger.error(f"Error in main loop: {e}")
                    print("J.A.S.O.N.: I encountered an error. Please check the logs.")

        except Exception as e:
            logger.critical(f"Critical error: {e}", exc_info=True)
            print("J.A.S.O.N.: A critical error occurred. Please check the logs.")
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources before shutdown."""
        if hasattr(self, 'vision'):
            self.vision.stop_continuous_vision()
        logger.info("Shutting down J.A.S.O.N.")
        # Add cleanup code here

def create_config_template():
    """Create a template configuration file."""
    template = """# J.A.S.O.N. Configuration Template

# General Settings
name: "J.A.S.O.N."
version: "0.1.0"
language: "en-US"

# Voice Settings
voice_input:
  enabled: false
  sensitivity: 0.5
  timeout: 5

voice_output:
  enabled: false
  rate: 200
  volume: 1.0
  elevenlabs_voice: "Adam"

# Vision Settings
vision:
  enabled: false
  capture_interval: 10  # seconds

# API Keys (store sensitive data in .env file)
api_keys:
  gemini: ""  # For vision analysis
  elevenlabs: ""  # For high-quality TTS
  openai: ""  # For general AI tasks

# Swarm Configuration
swarm:
  max_agents: 5
  default_timeout: 30

# Hologram Settings
hologram:
  enabled: true
  port: 5000
  host: "0.0.0.0"

# Memory Settings
memory:
  persist_path: "./jason_memory"
  max_results: 5

# Skill Development
skills:
  auto_save: true
  test_scripts: true

# Paths
paths:
  logs: "logs/"
  data: "data/"
  cache: "cache/"
  skills: "jason_skills/"

# Logging
logging:
  level: "INFO"
  file: "jason.log"
  max_size: 10
  backup_count: 5
"""
    config_path = Path("config.yaml")
    if config_path.exists():
        print(f"Configuration file already exists at {config_path}")
        return

    with open(config_path, 'w') as f:
        f.write(template)
    print(f"Created configuration template at {config_path}")

@click.group()
def cli():
    """J.A.S.O.N. command line interface."""
    pass

@cli.command()
def run():
    """Start J.A.S.O.N. in interactive mode."""
    # Load environment variables from .env file if it exists
    load_dotenv()

    # Initialize and start J.A.S.O.N.
    jason = JASON()
    jason.start()

@cli.command()
@click.option('--path', default='config.yaml', help='Path to save the configuration template')
def init_config(path):
    """Create a configuration template file."""
    create_config_template()

@cli.command()
@click.option('--host', default='127.0.0.1', help='Host to bind the API server to')
@click.option('--port', default=8000, type=int, help='Port to bind the API server to')
def api(host, port):
    """Start the FastAPI API server."""
    from jason.core.api_server import start_api_server
    click.echo(f"Starting J.A.S.O.N. API server on {host}:{port}")
    start_api_server(host=host, port=port)
    """Control VPN connections and privacy drift"""
    from jason.modules.cyber_stealth import CyberStealthManager

    # Load config
    load_dotenv()
    jason = JASON()
    manager = CyberStealthManager(jason.config, getattr(jason, 'hologram', None))

    if disconnect:
        result = manager.disconnect_vpn()
        click.echo(f"VPN disconnect: {'Success' if result['success'] else 'Failed'}")
    elif drift:
        if manager.drift_active:
            result = manager.stop_privacy_drift()
        else:
            result = manager.start_privacy_drift()
        click.echo(f"Privacy drift: {result['message']}")
    elif country:
        result = manager.connect_vpn(country)
        click.echo(f"VPN connect to {country}: {'Success' if result['success'] else 'Failed'}")
    else:
        status = manager.get_status()
        click.echo("VPN Status:")
        click.echo(f"  Connected: {status['vpn_connected']}")
        click.echo(f"  Country: {status['vpn_country']}")
        click.echo(f"  Privacy Drift: {status['privacy_drift_active']}")
        click.echo(f"  Rotation Interval: {status['rotation_interval_hours']} hours")

if __name__ == "__main__":
    cli()
