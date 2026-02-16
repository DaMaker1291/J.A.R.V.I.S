"""
J.A.S.O.N. Kinetic Holographic HUD - Real-Time Volumetric Visualization
"""

from flask import Flask, render_template_string
from flask_socketio import SocketIO, emit
import threading
import time
import json
from typing import Dict, Any
from jason.core.overlay import play_chirp

# JARVIS Aesthetic Protocol - Stark Colors
STARK_CYAN = "#00f2ff"
ALERT_RED = "#ff3c00"
NEUTRAL_ORANGE = "#ff8c00"

class HologramManager:
    def __init__(self):
        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")

        # Current status
        self.status = "idle"
        self.mood = "blue"  # blue=normal, red=danger, orange=warning
        self.threat_level = 0.0

        # Setup routes
        self._setup_routes()

        # Start server in background
        self.server_thread = threading.Thread(target=self._run_server)
        self.server_thread.daemon = True
        self.server_thread.start()

    def _setup_routes(self):
        """Setup Flask routes"""

        @self.app.route('/')
        def index():
            return render_template_string(self._get_html_template())

        @self.socketio.on('connect')
        def handle_connect():
            emit('status_update', {
                'status': self.status,
                'mood': self.mood,
                'threat_level': self.threat_level
            })

    def _get_html_template(self) -> str:
        """Get the HTML template for the hologram HUD"""
        return """
<!DOCTYPE html>
<html>
<head>
    <title>J.A.S.O.N. Holographic HUD</title>
    <style>
        body {
            margin: 0;
            padding: 0;
            background: black;
            overflow: hidden;
            font-family: 'Courier New', monospace;
        }
        .hologram {
            position: absolute;
            width: 100%;
            height: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .pyramid {
            width: 400px;
            height: 400px;
            position: relative;
            transform-style: preserve-3d;
        }
        .face {
            position: absolute;
            width: 0;
            height: 0;
            border-left: 200px solid transparent;
            border-right: 200px solid transparent;
            border-bottom: 346px solid rgba(0, 255, 255, 0.3);
            transform-origin: 50% 100%;
        }
        .face:nth-child(1) { transform: rotateY(0deg) translateZ(200px); }
        .face:nth-child(2) { transform: rotateY(90deg) translateZ(200px); }
        .face:nth-child(3) { transform: rotateY(180deg) translateZ(200px); }
        .face:nth-child(4) { transform: rotateY(270deg) translateZ(200px); }
        .face.top {
            border-left: 200px solid transparent;
            border-right: 200px solid transparent;
            border-bottom: 346px solid rgba(0, 255, 255, 0.3);
            transform: rotateX(90deg) translateZ(173px);
        }
        .status-text {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            color: #00ffff;
            font-size: 24px;
            text-align: center;
            z-index: 10;
        }
        .mood-indicator {
            position: absolute;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            width: 100px;
            height: 20px;
            background: rgba(0, 255, 255, 0.5);
            border-radius: 10px;
        }
        .threat-level {
            position: absolute;
            top: 20px;
            right: 20px;
            color: #ff0000;
            font-size: 18px;
        }
    </style>
</head>
<body>
    <div class="hologram">
        <div class="pyramid">
            <div class="face"></div>
            <div class="face"></div>
            <div class="face"></div>
            <div class="face"></div>
            <div class="face top"></div>
        </div>
        <div class="status-text" id="status">J.A.S.O.N. Online</div>
        <div class="mood-indicator" id="mood"></div>
        <div class="threat-level" id="threat">Threat: 0.0</div>
    </div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
    <script>
        const socket = io();
        const statusDiv = document.getElementById('status');
        const moodDiv = document.getElementById('mood');
        const threatDiv = document.getElementById('threat');

        socket.on('status_update', function(data) {
            statusDiv.textContent = data.status.toUpperCase();
            moodDiv.style.background = getMoodColor(data.mood);
            threatDiv.textContent = `Threat: ${data.threat_level}`;

            // Animate based on status
            animateHologram(data.status);
        });

        function getMoodColor(mood) {
            switch(mood) {
                case 'blue': return 'rgba(0, 255, 255, 0.5)';
                case 'red': return 'rgba(255, 0, 0, 0.5)';
                case 'orange': return 'rgba(255, 165, 0, 0.5)';
                default: return 'rgba(0, 255, 255, 0.5)';
            }
        }

        function animateHologram(status) {
            const pyramid = document.querySelector('.pyramid');
            pyramid.style.animation = '';

            switch(status) {
                case 'searching':
                    pyramid.style.animation = 'pulse 1s infinite';
                    break;
                case 'processing':
                    pyramid.style.animation = 'rotate 2s infinite linear';
                    break;
                case 'completed':
                    pyramid.style.animation = 'glow 0.5s ease-in-out';
                    break;
                default:
                    pyramid.style.animation = '';
            }
        }

        // Add CSS animations
        const style = document.createElement('style');
        style.textContent = `
            @keyframes pulse {
                0% { transform: scale(1); }
                50% { transform: scale(1.05); }
                100% { transform: scale(1); }
            }
            @keyframes rotate {
                from { transform: rotateY(0deg); }
                to { transform: rotateY(360deg); }
            }
            @keyframes glow {
                0% { filter: brightness(1); }
                50% { filter: brightness(1.5); }
                100% { filter: brightness(1); }
            }
        `;
        document.head.appendChild(style);
    </script>
</body>
</html>
        """

    def _run_server(self):
        """Run the Flask server"""
        self.socketio.run(self.app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)

    def send_status(self, status: str, threat_level: float = 0.0):
        """Send status update to hologram"""
        self.status = status
        self.threat_level = threat_level

        # Update mood based on status and threat
        if threat_level > 0.7 or status in ["danger", "alert"]:
            self.mood = "red"
            play_chirp("alert")  # Alert chirp
        elif threat_level > 0.3 or status in ["warning", "processing"]:
            self.mood = "orange"
            if status == "processing":
                play_chirp("processing")  # Low-frequency mechanical hum when processing
            else:
                play_chirp("warning")  # Warning chirp
        else:
            self.mood = "blue"
            if status == "success":
                play_chirp("confirm")  # Confirmation chirp
            elif status == "spatial_anchor":
                play_chirp("spatial_anchor")  # Sharp chirps when spatial anchor located

        data = {
            "status": self.status,
            "mood": self.mood,
            "threat_level": self.threat_level,
            "timestamp": time.time()
        }

        self.socketio.emit('status_update', data)

    def send_telemetry(self, telemetry_data: Dict[str, Any]):
        """Send telemetry data to hologram interface"""
        self.socketio.emit('telemetry', telemetry_data)
