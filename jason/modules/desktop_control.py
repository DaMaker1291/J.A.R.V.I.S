"""
J.A.S.O.N. Desktop Control Module
General Purpose Desktop Automation using PyAutoGUI
"""

import pyautogui
import time
import random
import logging
from typing import Dict, Any, Optional, Tuple
import subprocess
import platform

logger = logging.getLogger(__name__)

class DesktopController:
    """Desktop automation controller using PyAutoGUI"""

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.safety_bounds = self.config.get('safety_bounds', {
            'min_x': 100, 'max_x': 1800,
            'min_y': 100, 'max_y': 1000
        })

        # Configure PyAutoGUI
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.5

    def move_mouse(self, x: int, y: int, duration: float = 0.5) -> bool:
        """Move mouse to position with human-like movement"""
        try:
            # Add slight randomness for human-like movement
            x += random.randint(-5, 5)
            y += random.randint(-5, 5)

            # Ensure within safety bounds
            x = max(self.safety_bounds['min_x'], min(x, self.safety_bounds['max_x']))
            y = max(self.safety_bounds['min_y'], min(y, self.safety_bounds['max_y']))

            pyautogui.moveTo(x, y, duration=duration)
            logger.info(f"Moved mouse to ({x}, {y})")
            return True
        except Exception as e:
            logger.error(f"Mouse move failed: {e}")
            return False

    def click(self, x: Optional[int] = None, y: Optional[int] = None, button: str = 'left') -> bool:
        """Click at position or current position"""
        try:
            if x is not None and y is not None:
                self.move_mouse(x, y)

            pyautogui.click(button=button)
            logger.info(f"Clicked {button} button")
            return True
        except Exception as e:
            logger.error(f"Click failed: {e}")
            return False

    def double_click(self, x: Optional[int] = None, y: Optional[int] = None) -> bool:
        """Double click at position"""
        try:
            if x is not None and y is not None:
                self.move_mouse(x, y)

            pyautogui.doubleClick()
            logger.info("Double clicked")
            return True
        except Exception as e:
            logger.error(f"Double click failed: {e}")
            return False

    def right_click(self, x: Optional[int] = None, y: Optional[int] = None) -> bool:
        """Right click at position"""
        try:
            if x is not None and y is not None:
                self.move_mouse(x, y)

            pyautogui.rightClick()
            logger.info("Right clicked")
            return True
        except Exception as e:
            logger.error(f"Right click failed: {e}")
            return False

    def type_text(self, text: str, human_like: bool = True) -> bool:
        """Type text with optional human-like delays"""
        try:
            if human_like:
                for char in text:
                    pyautogui.typewrite(char)
                    # Random delay between 0.05-0.15 seconds
                    time.sleep(random.uniform(0.05, 0.15))
                    # Occasional longer pause (thinking)
                    if random.random() < 0.1:
                        time.sleep(random.uniform(0.2, 0.5))
            else:
                pyautogui.typewrite(text)

            logger.info(f"Typed text: {text[:50]}...")
            return True
        except Exception as e:
            logger.error(f"Typing failed: {e}")
            return False

    def press_key(self, key: str, presses: int = 1) -> bool:
        """Press a key multiple times"""
        try:
            for _ in range(presses):
                pyautogui.press(key)
                time.sleep(0.1)
            logger.info(f"Pressed {key} {presses} times")
            return True
        except Exception as e:
            logger.error(f"Key press failed: {e}")
            return False

    def hotkey(self, *keys: str) -> bool:
        """Press hotkey combination"""
        try:
            pyautogui.hotkey(*keys)
            logger.info(f"Pressed hotkey: {'+'.join(keys)}")
            return True
        except Exception as e:
            logger.error(f"Hotkey failed: {e}")
            return False

    def scroll(self, clicks: int, x: Optional[int] = None, y: Optional[int] = None) -> bool:
        """Scroll mouse wheel"""
        try:
            if x is not None and y is not None:
                self.move_mouse(x, y)

            pyautogui.scroll(clicks)
            logger.info(f"Scrolled {clicks} clicks")
            return True
        except Exception as e:
            logger.error(f"Scroll failed: {e}")
            return False

    def take_screenshot(self, region: Optional[Tuple[int, int, int, int]] = None) -> Optional[str]:
        """Take screenshot of screen or region"""
        try:
            screenshot = pyautogui.screenshot(region=region)
            # Save to temp file
            import tempfile
            import os
            fd, path = tempfile.mkstemp(suffix='.png')
            os.close(fd)
            screenshot.save(path)
            logger.info(f"Screenshot saved to {path}")
            return path
        except Exception as e:
            logger.error(f"Screenshot failed: {e}")
            return None

    def locate_image(self, image_path: str, confidence: float = 0.8) -> Optional[Tuple[int, int, int, int]]:
        """Locate image on screen"""
        try:
            location = pyautogui.locateOnScreen(image_path, confidence=confidence)
            if location:
                logger.info(f"Image found at {location}")
                return location
            else:
                logger.info("Image not found")
                return None
        except Exception as e:
            logger.error(f"Image location failed: {e}")
            return None

    def open_application(self, app_name: str) -> bool:
        """Open application by name"""
        try:
            system = platform.system().lower()
            if system == 'darwin':  # macOS
                subprocess.run(['open', '-a', app_name], check=True)
            elif system == 'windows':
                subprocess.run(['start', app_name], shell=True, check=True)
            elif system == 'linux':
                subprocess.run([app_name], check=True)
            else:
                logger.error(f"Unsupported platform: {system}")
                return False

            logger.info(f"Opened application: {app_name}")
            time.sleep(2)  # Wait for app to open
            return True
        except Exception as e:
            logger.error(f"Failed to open {app_name}: {e}")
            return False

    def close_application(self, app_name: str) -> bool:
        """Close application by name (basic implementation)"""
        try:
            system = platform.system().lower()
            if system == 'darwin':
                subprocess.run(['pkill', '-f', app_name], check=True)
            elif system == 'windows':
                subprocess.run(['taskkill', '/f', '/im', f'{app_name}.exe'], check=True)
            elif system == 'linux':
                subprocess.run(['pkill', app_name], check=True)

            logger.info(f"Closed application: {app_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to close {app_name}: {e}")
            return False

    def get_mouse_position(self) -> Tuple[int, int]:
        """Get current mouse position"""
        return pyautogui.position()

    def get_screen_size(self) -> Tuple[int, int]:
        """Get screen size"""
        return pyautogui.size()

    def wait_for_image(self, image_path: str, timeout: int = 10, confidence: float = 0.8) -> bool:
        """Wait for image to appear on screen"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.locate_image(image_path, confidence):
                return True
            time.sleep(0.5)
        return False

    def execute_macro(self, macro: list) -> bool:
        """Execute a sequence of desktop actions"""
        try:
            for action in macro:
                action_type = action.get('type')
                params = action.get('params', {})

                if action_type == 'move':
                    self.move_mouse(**params)
                elif action_type == 'click':
                    self.click(**params)
                elif action_type == 'type':
                    self.type_text(**params)
                elif action_type == 'press':
                    self.press_key(**params)
                elif action_type == 'hotkey':
                    self.hotkey(*params.get('keys', []))
                elif action_type == 'wait':
                    time.sleep(params.get('seconds', 1))
                elif action_type == 'open_app':
                    self.open_application(params.get('name', ''))
                elif action_type == 'close_app':
                    self.close_application(params.get('name', ''))
                else:
                    logger.warning(f"Unknown macro action: {action_type}")

            logger.info("Macro executed successfully")
            return True
        except Exception as e:
            logger.error(f"Macro execution failed: {e}")
            return False
