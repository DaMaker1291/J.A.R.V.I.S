"""
J.A.S.O.N. VPN Control Tool
Cyber-Stealth Module - Privacy Drift Automation
"""

import subprocess
import logging
import time
from typing import Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)

class VPNController:
    """Controls VPN connections using system CLI tools"""

    def __init__(self, vpn_provider: str = "nordvpn"):
        """Initialize VPN controller

        Args:
            vpn_provider: VPN provider CLI tool (nordvpn, mullvad, etc.)
        """
        self.vpn_provider = vpn_provider
        self.connected_country: Optional[str] = None

    def connect(self, country: str) -> bool:
        """Connect to VPN in specified country

        Args:
            country: Country code (e.g., 'us', 'uk', 'de')

        Returns:
            bool: True if successful
        """
        try:
            cmd = [self.vpn_provider, "connect", country]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                self.connected_country = country
                logger.info(f"Connected to VPN in {country}")
                return True
            else:
                logger.error(f"Failed to connect to VPN: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            logger.error("VPN connection timed out")
            return False
        except Exception as e:
            logger.error(f"VPN connection error: {e}")
            return False

    def disconnect(self) -> bool:
        """Disconnect from VPN

        Returns:
            bool: True if successful
        """
        try:
            cmd = [self.vpn_provider, "disconnect"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

            if result.returncode == 0:
                self.connected_country = None
                logger.info("Disconnected from VPN")
                return True
            else:
                logger.error(f"Failed to disconnect VPN: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"VPN disconnect error: {e}")
            return False

    def get_status(self) -> dict:
        """Get current VPN status

        Returns:
            dict: Status information
        """
        try:
            cmd = [self.vpn_provider, "status"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)

            return {
                "connected": self.connected_country is not None,
                "country": self.connected_country,
                "output": result.stdout,
                "error": result.stderr
            }

        except Exception as e:
            logger.error(f"VPN status error: {e}")
            return {"connected": False, "country": None, "error": str(e)}

    def rotate_location(self, countries: List[str]) -> bool:
        """Rotate VPN location randomly from available countries

        Args:
            countries: List of country codes to rotate between

        Returns:
            bool: True if rotation successful
        """
        import random

        if not countries:
            logger.error("No countries provided for rotation")
            return False

        # Disconnect first
        self.disconnect()

        # Wait a moment
        time.sleep(2)

        # Connect to random country
        new_country = random.choice(countries)
        return self.connect(new_country)

    def start_privacy_drift(self, countries: List[str], interval_hours: int = 4):
        """Start automatic privacy drift rotation

        Args:
            countries: List of countries to rotate between
            interval_hours: Hours between rotations
        """
        import threading

        def drift_loop():
            while True:
                self.rotate_location(countries)
                time.sleep(interval_hours * 3600)

        drift_thread = threading.Thread(target=drift_loop, daemon=True)
        drift_thread.start()
        logger.info(f"Started privacy drift with {interval_hours}h intervals")
