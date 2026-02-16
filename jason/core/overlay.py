"""
J.A.S.O.N. Screen Overlay - PyQt6 for visual selection brackets
"""

import sys
from PyQt6.QtWidgets import QApplication, QWidget, QLabel
from PyQt6.QtGui import QPainter, QPen, QColor, QFont
from PyQt6.QtCore import Qt, QRect, QPoint
from typing import List, Dict, Any, Optional
import time
import os

# JARVIS Aesthetic Protocol - Stark Colors
STARK_CYAN = "#00f2ff"
ALERT_RED = "#ff3c00"
NEUTRAL_ORANGE = "#ff8c00"

class ScreenOverlay(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)

        # Get screen geometry
        screen = QApplication.primaryScreen().geometry()
        self.setGeometry(screen)

        self.rectangles = []  # List of {"rect": QRect, "label": str, "selected": bool}
        self.selected_callback = None
        self.hide()

    def show_overlay(self, rectangles_data: List[Dict[str, Any]], callback: Optional[callable] = None):
        """Show overlay with highlighted rectangles"""
        self.rectangles = []
        for rect_data in rectangles_data:
            rect = QRect(
                rect_data["x"],
                rect_data["y"],
                rect_data["w"],
                rect_data["h"]
            )
            self.rectangles.append({
                "rect": rect,
                "label": rect_data.get("label", ""),
                "selected": False
            })

        self.selected_callback = callback
        self.show()
        self.raise_()
        self.update()

        # Auto-hide after 10 seconds if no selection
        self.timer = self.startTimer(10000)

    def timerEvent(self, event):
        """Auto-hide timer"""
        self.hide()
        self.killTimer(self.timer)

    def paintEvent(self, event):
        """Draw the overlay"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Semi-transparent background
        painter.fillRect(self.rect(), QColor(0, 0, 0, 50))

        # Draw rectangles
        cyan_pen = QPen(QColor(STARK_CYAN), 3, Qt.PenStyle.SolidLine)  # Stark Cyan
        red_pen = QPen(QColor(ALERT_RED), 3, Qt.PenStyle.SolidLine)   # Alert Red
        orange_pen = QPen(QColor(NEUTRAL_ORANGE), 3, Qt.PenStyle.SolidLine)  # Neutral Orange

        font = QFont("Arial", 12, QFont.Weight.Bold)

        for i, item in enumerate(self.rectangles):
            rect = item["rect"]
            label = item["label"]

            # Use cyan for normal, red for selected
            pen = red_pen if item["selected"] else cyan_pen
            painter.setPen(pen)

            # Draw rectangle
            painter.drawRect(rect)

            # Draw pulsing effect for first rectangle (suggested selection)
            if i == 0 and not any(r["selected"] for r in self.rectangles):
                # Simple pulse effect
                pulse_intensity = int((time.time() * 2) % 2 * 100) + 155
                pulse_pen = QPen(QColor("#00f2ff").lighter(pulse_intensity), 5)
                painter.setPen(pulse_pen)
                painter.drawRect(rect.adjusted(-2, -2, 2, 2))

            # Draw label
            painter.setFont(font)
            painter.setPen(QColor(STARK_CYAN))  # Stark Cyan text
            label_rect = QRect(rect.x(), rect.y() - 25, rect.width(), 20)
            painter.drawText(label_rect, Qt.AlignmentFlag.AlignCenter, label)

    def mousePressEvent(self, event):
        """Handle mouse clicks for selection"""
        pos = event.pos()

        for item in self.rectangles:
            if item["rect"].contains(pos):
                # Select this rectangle
                for r in self.rectangles:
                    r["selected"] = False
                item["selected"] = True
                self.update()

                # Call callback if provided
                if self.selected_callback:
                    self.selected_callback(item["label"])

                # Hide overlay after selection
                self.hide()
                break

    def keyPressEvent(self, event):
        """Handle keyboard input"""
        if event.key() == Qt.Key.Key_Escape:
            self.hide()

        # Number keys for quick selection
        key = event.key()
        if Qt.Key.Key_1 <= key <= Qt.Key.Key_9:
            index = key - Qt.Key.Key_1
            if index < len(self.rectangles):
                item = self.rectangles[index]
                for r in self.rectangles:
                    r["selected"] = False
                item["selected"] = True
                self.update()

                if self.selected_callback:
                    self.selected_callback(item["label"])

                self.hide()

def play_chirp(chirp_type: str):
    """Play JARVIS aesthetic confirmation chirps"""
    try:
        if chirp_type == "confirm":
            # Double-beep for confirmation
            os.system("afplay /System/Library/Sounds/Glass.aiff")
            time.sleep(0.2)
            os.system("afplay /System/Library/Sounds/Glass.aiff")
        elif chirp_type == "alert":
            # Alert sound
            os.system("afplay /System/Library/Sounds/Basso.aiff")
        elif chirp_type == "warning":
            # Warning sound
            os.system("afplay /System/Library/Sounds/Ping.aiff")
        elif chirp_type == "waiting":
            # Low-frequency hum (simple beep)
            os.system("afplay /System/Library/Sounds/Tink.aiff")
        elif chirp_type == "processing":
            # Low-frequency mechanical hums when processing
            # Play a longer, lower-frequency sound
            os.system("afplay /System/Library/Sounds/Purr.aiff")  # Low frequency hum
        elif chirp_type == "spatial_anchor":
            # Sharp chirps when spatial anchor is located
            # Quick, sharp chirp sequence
            for i in range(3):
                os.system("afplay /System/Library/Sounds/Pop.aiff")
                time.sleep(0.1)
    except:
        # Silent fail if sounds not available
        pass

def highlight_screen_areas(rectangles: List[Dict[str, Any]], callback: Optional[callable] = None):
    """Convenience function to highlight areas on screen"""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    overlay = ScreenOverlay()
    overlay.show_overlay(rectangles, callback)

    # If no app was running, start event loop
    if QApplication.instance() == app:
        app.exec()

    return overlay
