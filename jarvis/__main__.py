#!/usr/bin/env python3
"""
J.A.R.V.I.S. - Just A Rather Very Intelligent System
A personal assistant inspired by the AI from the Iron Man series.
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

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("jarvis.log")
    ]
)
logger = logging.getLogger(__name__)

class JARVIS:
    """Main J.A.R.V.I.S. class that orchestrates all functionality."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize J.A.R.V.I.S. with configuration."""
        self.config = self._load_config(config_path)
        self._initialize_components()
        logger.info(f"Initialized {self.config.get('name', 'J.A.R.V.I.S.')} v{self.config.get('version', '0.1.0')}")
    
    def _load_config(self, config_path: Optional[str] = None) -> dict:
        """Load configuration from YAML file."""
        if config_path is None:
            config_path = Path(__file__).parent / "config" / "config.yaml"
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f) or {}
            
            # Ensure required directories exist
            for path_key in ['logs', 'data', 'cache']:
                if path_key in config.get('paths', {}):
                    os.makedirs(config['paths'][path_key], exist_ok=True)
            
            return config
            
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return {}
    
    def _initialize_components(self):
        """Initialize all components based on configuration."""
        # This will be expanded with actual component initialization
        pass
    
    def start(self):
        """Start the J.A.R.V.I.S. main loop."""
        logger.info("Starting J.A.R.V.I.S. (Press Ctrl+C to exit)")
        print(f"\n{'='*50}")
        print(f"  {self.config.get('name', 'J.A.R.V.I.S.')} v{self.config.get('version', '0.1.0')}")
        print("  Type 'help' for available commands")
        print("  Press Ctrl+C to exit")
        print("="*50 + "\n")
        
        try:
            while True:
                try:
                    # For now, just a simple command loop
                    command = input("You: ").strip().lower()
                    
                    if not command:
                        continue
                        
                    if command in ['exit', 'quit', 'bye']:
                        print("J.A.R.V.I.S.: Goodbye!")
                        break
                        
                    # Process command (to be implemented)
                    print(f"J.A.R.V.I.S.: I heard: {command}")
                    
                except KeyboardInterrupt:
                    print("\nJ.A.R.V.I.S.: Shutting down...")
                    break
                except Exception as e:
                    logger.error(f"Error in main loop: {e}")
                    print("J.A.R.V.I.S.: I encountered an error. Please check the logs.")
                    
        except Exception as e:
            logger.critical(f"Critical error: {e}", exc_info=True)
            print("J.A.R.V.I.S.: A critical error occurred. Please check the logs.")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources before shutdown."""
        logger.info("Shutting down J.A.R.V.I.S.")
        # Add cleanup code here

def create_config_template():
    """Create a template configuration file."""
    template = """# J.A.R.V.I.S. Configuration Template

# General Settings
name: "J.A.R.V.I.S."
version: "0.1.0"
language: "en-US"

# Voice Settings
voice:
  rate: 150
  volume: 1.0
  voice_id: null

# Wake Word Detection
wake_word:
  enabled: true
  sensitivity: 0.5
  timeout: 5

# Modules
modules:
  weather: true
  news: true
  calendar: true
  email: false
  smart_home: false
  system_control: true
  web_search: true

# API Keys (store sensitive data in .env file)
api_keys:
  openai: ""
  weather: ""

# Paths
paths:
  logs: "logs/"
  data: "data/"
  cache: "cache/"

# Logging
logging:
  level: "INFO"
  file: "jarvis.log"
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
    """J.A.R.V.I.S. command line interface."""
    pass

@cli.command()
def run():
    """Start J.A.R.V.I.S. in interactive mode."""
    # Load environment variables from .env file if it exists
    load_dotenv()
    
    # Initialize and start J.A.R.V.I.S.
    jarvis = JARVIS()
    jarvis.start()

@cli.command()
@click.option('--path', default='config.yaml', help='Path to save the configuration template')
def init_config(path):
    """Create a configuration template file."""
    create_config_template()

if __name__ == "__main__":
    cli()
