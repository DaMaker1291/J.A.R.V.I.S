"""
J.A.S.O.N. Deep Sight - Multimodal Vision Module
Hyper-Spatial Vision & Vital Telemetry: Biometric Intent & Emotional Mapping
"""

import cv2
import pyautogui
import google.generativeai as genai
from langchain_core.tools import Tool
from PIL import Image
import numpy as np
import threading
import time
from typing import Dict, Any, List, Tuple, Optional
from scipy import signal
from scipy.fft import fft, ifft
from collections import deque
import math

# YOLOv8 for object detection
yolo_available = False
try:
    from ultralytics import YOLO
    yolo_available = True
except ImportError:
    pass

# MediaPipe for advanced facial analysis
mediapipe_available = False
try:
    import mediapipe as mp
    mediapipe_available = True
except ImportError:
    pass

class VisionManager:
    """Hyper-Spatial Vision & Vital Telemetry: Biometric Intent & Emotional Mapping"""

    def __init__(self, gemini_api_key: str = ""):
        if gemini_api_key:
            genai.configure(api_key=gemini_api_key)
            self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
            self.vision_available = True
        else:
            self.vision_available = False

        self.vision_active = False
        self.vision_thread = None

        # Enhanced biometric analysis
        self._initialize_biometric_systems()

        # Vital telemetry tracking
        self.vital_signs = {
            'heart_rate': 0,
            'respiratory_rate': 0,
            'stress_level': 0.0,
            'fatigue_level': 0.0,
            'deception_probability': 0.0
        }

        # Emotion tracking
        self.emotion_history = deque(maxlen=100)
        self.baseline_emotions = {}

        # Saccade and pupil tracking
        self.eye_tracking_data = {
            'pupil_sizes': deque(maxlen=300),  # Last 10 seconds at 30fps
            'eye_positions': deque(maxlen=300),
            'saccades': [],
            'micro_saccades': []
        }

        # Skin analysis for heart rate
        self.skin_roi_history = deque(maxlen=600)  # 20 seconds at 30fps
        self.heart_rate_buffer = []

        # YOLOv8 for object detection
        self.yolo_model = None
        if yolo_available:
            try:
                self.yolo_model = YOLO('yolov8n.pt')  # Load nano model for speed
            except Exception as e:
                print(f"YOLO loading failed: {e}")

    def _initialize_biometric_systems(self):
        """Initialize all biometric analysis systems"""
        # OpenCV classifiers
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

        # MediaPipe for advanced facial analysis
        if mediapipe_available:
            self.mp_face_mesh = mp.solutions.face_mesh
            self.mp_drawing = mp.solutions.drawing_utils
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        else:
            self.face_mesh = None

        # Camera properties for calibration
        self.camera_matrix = None
        self.dist_coeffs = None

    def analyze_biometrics(self, frame: np.ndarray) -> Dict[str, Any]:
        """Analyze biometric signals from camera frame"""
        results = {
            'face_detected': False,
            'eye_tracking': {},
            'vital_signs': {},
            'emotional_state': {},
            'deception_indicators': {}
        }

        # Detect face
        faces = self.face_cascade.detectMultiScale(frame, 1.3, 5)
        if len(faces) > 0:
            results['face_detected'] = True
            face = faces[0]  # Focus on primary face

            # Extract face region
            x, y, w, h = face
            face_roi = frame[y:y+h, x:x+w]

            # Analyze eyes and pupils
            eye_results = self._analyze_eye_tracking(face_roi, (x, y, w, h))
            results['eye_tracking'] = eye_results

            # Analyze skin color for heart rate
            hr_results = self._analyze_heart_rate(face_roi)
            results['vital_signs'] = hr_results

            # Analyze emotional state
            emotion_results = self._analyze_emotional_state(face_roi, eye_results)
            results['emotional_state'] = emotion_results

            # Detect deception indicators
            deception_results = self._detect_deception_indicators(eye_results, emotion_results)
            results['deception_indicators'] = deception_results

        return results

    def _analyze_eye_tracking(self, face_roi: np.ndarray, face_coords: Tuple[int, int, int, int]) -> Dict[str, Any]:
        """Analyze eye movements, pupil dilation, and saccades"""
        eye_results = {
            'pupil_dilation': 0.0,
            'saccade_detected': False,
            'micro_saccades': [],
            'eye_positions': (0, 0),
            'gaze_direction': 'center'
        }

        # Convert to grayscale for eye detection
        gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

        # Detect eyes
        eyes = self.eye_cascade.detectMultiScale(gray_face, 1.1, 3)

        if len(eyes) >= 1:
            # Focus on the most prominent eye for pupil analysis
            eye = eyes[0]
            ex, ey, ew, eh = eye
            eye_roi = gray_face[ey:ey+eh, ex:ex+ew]

            # Pupil detection (darkest region in eye)
            pupil_size, pupil_center = self._detect_pupil(eye_roi)

            if pupil_size > 0:
                eye_results['pupil_dilation'] = pupil_size

                # Convert to absolute coordinates
                abs_eye_x = face_coords[0] + ex + pupil_center[0]
                abs_eye_y = face_coords[1] + ey + pupil_center[1]

                # Track eye position for saccade detection
                current_pos = (abs_eye_x, abs_eye_y)
                self.eye_tracking_data['eye_positions'].append(current_pos)
                eye_results['eye_positions'] = current_pos

                # Detect saccades
                saccade_info = self._detect_saccades(current_pos)
                eye_results.update(saccade_info)

        return eye_results

    def _detect_pupil(self, eye_roi: np.ndarray) -> Tuple[float, Tuple[int, int]]:
        """Detect pupil in eye region using image processing"""
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(eye_roi, (7, 7), 0)

        # Threshold to find dark regions (pupil)
        _, threshold = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY_INV)

        # Find contours
        contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Find largest contour (likely pupil)
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)

            if area > 10:  # Minimum pupil size
                # Calculate centroid
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    return area, (cx, cy)

        return 0.0, (0, 0)

    def _detect_saccades(self, current_pos: Tuple[int, int]) -> Dict[str, Any]:
        """Detect saccades and micro-saccades in eye movement"""
        saccade_results = {
            'saccade_detected': False,
            'micro_saccades': []
        }

        if len(self.eye_tracking_data['eye_positions']) < 2:
            return saccade_results

        # Calculate movement velocity
        prev_pos = self.eye_tracking_data['eye_positions'][-2]
        dx = current_pos[0] - prev_pos[0]
        dy = current_pos[1] - prev_pos[1]
        distance = math.sqrt(dx*dx + dy*dy)

        # Saccade detection (rapid eye movement > 2 degrees visual angle)
        # Approximate: > 30 pixels rapid movement
        if distance > 30:
            saccade_results['saccade_detected'] = True

        # Micro-saccade detection (smaller rapid movements)
        elif distance > 5 and distance <= 15:
            saccade_results['micro_saccades'].append({
                'timestamp': time.time(),
                'magnitude': distance,
                'direction': math.atan2(dy, dx)
            })

        return saccade_results

    def _analyze_heart_rate(self, face_roi: np.ndarray) -> Dict[str, Any]:
        """Analyze heart rate from skin color fluctuations (Eulerian method)"""
        hr_results = {
            'heart_rate': 0,
            'confidence': 0.0,
            'signal_quality': 'poor'
        }

        # Extract forehead region (green channel for blood volume)
        h, w = face_roi.shape[:2]
        forehead_roi = face_roi[int(h*0.1):int(h*0.3), int(w*0.3):int(w*0.7)]

        if forehead_roi.size > 0:
            # Convert to YUV color space for better blood volume detection
            yuv = cv2.cvtColor(forehead_roi, cv2.COLOR_BGR2YUV)

            # Use green channel (most sensitive to blood volume changes)
            green_channel = yuv[:, :, 1].astype(float)

            # Calculate average green intensity
            avg_green = np.mean(green_channel)
            self.skin_roi_history.append(avg_green)

            # Need enough data for FFT analysis
            if len(self.skin_roi_history) >= 150:  # ~5 seconds at 30fps
                # Detrend signal
                signal = np.array(list(self.skin_roi_history))
                detrended = signal - np.mean(signal)

                # Bandpass filter (0.5-3.0 Hz for heart rate 30-180 BPM)
                from scipy.signal import butter, filtfilt
                b, a = butter(2, [0.5/15, 3.0/15], btype='band')  # 15fps sampling
                filtered = filtfilt(b, a, detrended)

                # FFT analysis
                fft_result = np.abs(fft(filtered))
                freqs = np.fft.fftfreq(len(filtered), 1/30)  # 30fps

                # Find peak in heart rate range (0.5-3.0 Hz = 30-180 BPM)
                hr_range = (freqs >= 0.5) & (freqs <= 3.0)
                if np.any(hr_range):
                    peak_idx = np.argmax(fft_result[hr_range])
                    peak_freq = freqs[hr_range][peak_idx]

                    # Convert frequency to BPM
                    heart_rate = peak_freq * 60
                    confidence = fft_result[hr_range][peak_idx] / np.max(fft_result)

                    hr_results.update({
                        'heart_rate': int(heart_rate),
                        'confidence': float(confidence),
                        'signal_quality': 'good' if confidence > 0.3 else 'fair'
                    })

        return hr_results

    def _analyze_emotional_state(self, face_roi: np.ndarray, eye_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze emotional state from facial features and eye data"""
        emotion_results = {
            'stress_level': 0.0,
            'fatigue_level': 0.0,
            'emotional_state': 'neutral',
            'confidence': 0.0
        }

        # Convert to grayscale for analysis
        gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

        # Detect smiles (positive emotion indicator)
        smiles = self.smile_cascade.detectMultiScale(gray_face, 1.8, 20)

        # Pupil dilation analysis (stress/fatigue indicator)
        pupil_size = eye_data.get('pupil_dilation', 0)

        # Calculate baseline if not established
        if not self.baseline_emotions:
            self.baseline_emotions = {
                'avg_pupil_size': pupil_size,
                'samples': 1
            }
        else:
            # Update baseline with exponential moving average
            alpha = 0.1
            self.baseline_emotions['avg_pupil_size'] = (
                alpha * pupil_size +
                (1 - alpha) * self.baseline_emotions['avg_pupil_size']
            )
            self.baseline_emotions['samples'] += 1

        # Analyze pupil dilation relative to baseline
        if self.baseline_emotions['samples'] > 10:
            baseline_pupil = self.baseline_emotions['avg_pupil_size']
            dilation_ratio = pupil_size / baseline_pupil if baseline_pupil > 0 else 1.0

            # Stress indicators: dilated pupils, rapid micro-saccades
            stress_indicators = 0
            if dilation_ratio > 1.2:  # Dilated pupils
                stress_indicators += 0.3
            if len(eye_data.get('micro_saccades', [])) > 3:  # Frequent micro-saccades
                stress_indicators += 0.4

            # Fatigue indicators: contracted pupils, reduced blink rate
            fatigue_indicators = 0
            if dilation_ratio < 0.8:  # Contracted pupils
                fatigue_indicators += 0.3

            emotion_results.update({
                'stress_level': min(stress_indicators, 1.0),
                'fatigue_level': min(fatigue_indicators, 1.0),
                'emotional_state': self._classify_emotion(smiles, stress_indicators, fatigue_indicators),
                'confidence': 0.7  # Base confidence for analysis
            })

        return emotion_results

    def _classify_emotion(self, smiles: np.ndarray, stress: float, fatigue: float) -> str:
        """Classify emotional state based on facial features"""
        if len(smiles) > 0:
            return 'happy'
        elif stress > 0.6:
            return 'stressed'
        elif fatigue > 0.6:
            return 'fatigued'
        elif stress > 0.3:
            return 'anxious'
        else:
            return 'neutral'

    def _detect_deception_indicators(self, eye_data: Dict[str, Any], emotion_data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect deception indicators from eye movements and emotional state"""
        deception_results = {
            'deception_probability': 0.0,
            'indicators': []
        }

        probability = 0.0

        # Eye contact avoidance (frequent saccades away from camera)
        if eye_data.get('saccade_detected', False):
            probability += 0.2
            deception_results['indicators'].append('avoided_eye_contact')

        # Micro-saccades (increased cognitive load from lying)
        micro_saccade_count = len(eye_data.get('micro_saccades', []))
        if micro_saccade_count > 5:
            probability += 0.3
            deception_results['indicators'].append('increased_micro_saccades')

        # Stress indicators (physiological response to deception)
        stress_level = emotion_data.get('stress_level', 0)
        if stress_level > 0.5:
            probability += 0.2
            deception_results['indicators'].append('elevated_stress')

        # Pupil dilation (cognitive load)
        pupil_dilation = eye_data.get('pupil_dilation', 0)
        if self.baseline_emotions and pupil_dilation > self.baseline_emotions['avg_pupil_size'] * 1.3:
            probability += 0.3
            deception_results['indicators'].append('pupil_dilation')

        deception_results['deception_probability'] = min(probability, 1.0)

        return deception_results

    def get_vital_telemetry(self) -> Dict[str, Any]:
        """Get current vital signs and biometric data"""
        return {
            'vital_signs': self.vital_signs.copy(),
            'emotional_state': self._get_current_emotion(),
            'biometric_status': {
                'face_tracked': len(self.emotion_history) > 0,
                'pupil_tracking_active': len(self.eye_tracking_data['pupil_sizes']) > 0,
                'heart_rate_monitoring': len(self.skin_roi_history) > 100
            }
        }

    def _get_current_emotion(self) -> Dict[str, Any]:
        """Get current emotional state summary"""
        if not self.emotion_history:
            return {'state': 'unknown', 'confidence': 0.0}

        # Average recent emotional states
        recent_emotions = list(self.emotion_history)[-10:]  # Last 10 readings

        avg_stress = np.mean([e['stress_level'] for e in recent_emotions])
        avg_fatigue = np.mean([e['fatigue_level'] for e in recent_emotions])

        # Determine dominant emotion
        if avg_stress > 0.6:
            state = 'highly_stressed'
        elif avg_fatigue > 0.6:
            state = 'highly_fatigued'
        elif avg_stress > 0.4:
            state = 'stressed'
        elif avg_fatigue > 0.4:
            state = 'fatigued'
        else:
            state = 'normal'

        return {
            'state': state,
            'stress_level': float(avg_stress),
            'fatigue_level': float(avg_fatigue),
            'confidence': 0.8
        }

    def start_continuous_vision(self):
        """Start continuous vision monitoring with biometric analysis"""
        if self.vision_active:
            return "Vision monitoring already active"

        try:
            self.vision_active = True
            self.vision_thread = threading.Thread(target=self._continuous_vision_loop, daemon=True)
            self.vision_thread.start()
            return "Continuous vision monitoring started with biometric analysis"
        except Exception as e:
            self.vision_active = False
            return f"Failed to start continuous vision: {e}"

    def stop_continuous_vision(self):
        """Stop continuous vision monitoring"""
        if not self.vision_active:
            return "Vision monitoring not active"

        self.vision_active = False
        if self.vision_thread:
            self.vision_thread.join(timeout=5)
        return "Continuous vision monitoring stopped"

    def _continuous_vision_loop(self):
        """Main loop for continuous vision monitoring with biometric analysis"""
        cap = None
        try:
            # Initialize webcam
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("Could not open webcam")
                return

            frame_count = 0
            last_biometric_update = time.time()

            while self.vision_active:
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.1)
                    continue

                frame_count += 1

                # Run biometric analysis every 3 frames (10fps) to avoid overloading
                if frame_count % 3 == 0:
                    biometric_results = self.analyze_biometrics(frame)

                    # Update vital signs
                    if biometric_results['face_detected']:
                        # Update heart rate
                        hr_data = biometric_results['vital_signs']
                        if hr_data['heart_rate'] > 0:
                            self.vital_signs['heart_rate'] = hr_data['heart_rate']

                        # Update emotional state
                        emotion_data = biometric_results['emotional_state']
                        self.emotion_history.append({
                            'stress_level': emotion_data['stress_level'],
                            'fatigue_level': emotion_data['fatigue_level'],
                            'timestamp': time.time()
                        })

                        # Update deception probability
                        deception_data = biometric_results['deception_indicators']
                        self.vital_signs['deception_probability'] = deception_data['deception_probability']

                    # Send updates every 2 seconds
                    current_time = time.time()
                    if current_time - last_biometric_update > 2.0:
                        self._send_biometric_updates(biometric_results)
                        last_biometric_update = current_time

                # Object detection every 10 frames (3fps)
                if frame_count % 10 == 0:
                    # Convert frame to PIL for object detection
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)
                    self._detect_objects(pil_image)

                time.sleep(0.03)  # ~30fps

        except Exception as e:
            print(f"Continuous vision loop error: {e}")
        finally:
            if cap:
                cap.release()
            self.vision_active = False

    def _send_biometric_updates(self, biometric_results: Dict[str, Any]):
        """Send biometric updates to the system"""
        # Update hologram with biometric status
        from jason.core.hologram import HologramManager
        hologram = HologramManager()

        if biometric_results['face_detected']:
            # Determine mood based on emotional state
            emotion = biometric_results['emotional_state']['emotional_state']
            stress = biometric_results['emotional_state']['stress_level']
            deception = biometric_results['deception_indicators']['deception_probability']

            # Set hologram status based on biometric state
            if deception > 0.7:
                hologram.send_status("caution", threat_level=0.8)
            elif stress > 0.6:
                hologram.send_status("stress_detected", threat_level=0.4)
            elif emotion == 'happy':
                hologram.send_status("positive", threat_level=0.0)
            else:
                hologram.send_status("monitoring", threat_level=0.1)

            # Log significant biometric events
            if biometric_results['eye_tracking']['saccade_detected']:
                print("Biometric Alert: Rapid eye movement detected")

            if deception > 0.5:
                print(f"Biometric Alert: High deception probability ({deception:.2f})")

            if biometric_results['vital_signs']['heart_rate'] > 100:
                print(f"Biometric Alert: Elevated heart rate ({biometric_results['vital_signs']['heart_rate']} BPM)")

        self.screen_capture_tool = Tool(
            name="Screen Capture",
            func=self.capture_screen,
            description="Capture current screen and analyze with vision AI"
        )

        self.webcam_capture_tool = Tool(
            name="Webcam Capture",
            func=self.capture_webcam,
            description="Capture webcam image and analyze with vision AI"
        )

        self.intent_analysis_tool = Tool(
            name="Intent Analysis",
            func=self.analyze_intent,
            description="Analyze visual context to understand user intent"
        )

        self.heart_rate_tool = Tool(
            name="Heart Rate Monitor",
            func=self.monitor_heart_rate,
            description="Monitor user's heart rate via webcam for stress detection"
        )

        self.pupil_dilation_tool = Tool(
            name="Pupil Dilation Monitor",
            func=self.monitor_pupil_dilation,
            description="Monitor pupil dilation for emotional state analysis"
        )

        self.saccade_tracking_tool = Tool(
            name="Saccade Tracking",
            func=self.monitor_saccades,
            description="Track rapid eye movements for focus and attention analysis"
        )

        self.spatial_anchor_tool = Tool(
            name="Spatial Anchor",
            func=self.find_object_location,
            description="Locate physical objects in the room using YOLOv8 object recognition"
        )

        self.personal_designer_tool = Tool(
            name="Personal Designer",
            func=self.sketch_to_openscad,
            description="Convert hand-drawn sketches to OpenSCAD code for 3D printing"
        )

    def start_continuous_vision(self):
        """Start continuous vision monitoring"""
        self.vision_active = True
        self.vision_thread = threading.Thread(target=self._vision_loop)
        self.vision_thread.start()

    def stop_continuous_vision(self):
        """Stop continuous vision monitoring"""
        self.vision_active = False
        if self.vision_thread:
            self.vision_thread.join()

    def _vision_loop(self):
        """Continuous vision monitoring loop"""
        while self.vision_active:
            try:
                # Capture screen and webcam
                screen_img = self._capture_screen_image()
                webcam_img = self._capture_webcam_image()

                # Analyze for intent
                intent = self.analyze_intent_from_images(screen_img, webcam_img)

                # Analyze hyper-spatial vision
                heart_rate = self._analyze_heart_rate(webcam_img)
                pupil_dilation = self._analyze_pupil_dilation(webcam_img)
                saccades = self._analyze_saccades(webcam_img)

                # Detect objects for spatial anchor
                self._detect_objects(webcam_img)

                # If intent or physiological changes detected, notify system
                if intent or heart_rate != "normal" or pupil_dilation != "normal" or saccades != "normal":
                    self._notify_system(intent, heart_rate, pupil_dilation, saccades)

                time.sleep(15)  # Check every 15 seconds as per spec
            except Exception as e:
                print(f"Vision loop error: {e}")

    def _capture_screen_image(self) -> Image.Image:
        """Capture screenshot"""
        screenshot = pyautogui.screenshot()
        return screenshot

    def _capture_webcam_image(self) -> Image.Image:
        """Capture webcam image"""
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()

        if ret:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return Image.fromarray(frame_rgb)
        else:
            # Return blank image if no webcam
            return Image.new('RGB', (640, 480), color='black')

    def capture_screen(self, description: str = "Analyze this screen capture") -> str:
        """Capture and analyze screen"""
        img = self._capture_screen_image()
        return self._analyze_image(img, description)

    def capture_webcam(self, description: str = "Analyze this webcam capture") -> str:
        """Capture and analyze webcam"""
        img = self._capture_webcam_image()
        return self._analyze_image(img, description)

    def _analyze_image(self, image: Image.Image, prompt: str) -> str:
        """Analyze image with Gemini"""
        if not self.vision_available:
            return "Vision analysis not available - no API key provided"

        try:
            response = self.model.generate_content([prompt, image])
            return response.text
        except Exception as e:
            return f"Vision analysis failed: {e}"

    def analyze_intent(self, context: str) -> str:
        """Analyze user intent from visual context"""
        prompt = f"""
        Analyze the following visual context and determine the user's likely intent or emotional state:

        {context}

        Provide a brief assessment of what the user might need or want based on this visual information.
        """

        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Intent analysis failed: {e}"

    def analyze_intent_from_images(self, screen_img: Image.Image, webcam_img: Image.Image) -> str:
        """Analyze intent from both screen and webcam"""
        combined_prompt = """
        Analyze both the screen content and the user's facial expression/webcam view.
        Determine the user's current intent, emotional state, or what assistance they might need.
        Look for signs of frustration, focus, or specific tasks being performed.
        """

        try:
            response = self.model.generate_content([combined_prompt, screen_img, webcam_img])
            return response.text
        except Exception as e:
            return f"Combined analysis failed: {e}"

    def _notify_system(self, intent: str, heart_rate: str, pupil_dilation: str, saccades: str):
        """Notify the main system of detected intent and physiological changes"""
        # This would integrate with the swarm manager
        notifications = []
        if intent:
            notifications.append(f"Intent: {intent}")
        if heart_rate != "normal":
            notifications.append(f"Heart rate: {heart_rate}")
        if pupil_dilation != "normal":
            notifications.append(f"Pupil dilation: {pupil_dilation}")
        if saccades != "normal":
            notifications.append(f"Saccades: {saccades}")

        if notifications:
            print(f"Hyper-spatial vision alerts: {'; '.join(notifications)}")
            # Could trigger proactive actions like dimming lights, playing music

    def _analyze_heart_rate(self, image: Image.Image) -> str:
        """Analyze heart rate using Eulerian Video Magnification"""
        try:
            # Convert PIL to OpenCV
            opencv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2GRAY)

            # Detect faces
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

            if len(faces) == 0:
                return "normal"

            # Use first face
            (x, y, w, h) = faces[0]

            # Extract forehead region (top 1/3 of face)
            forehead_y = int(y + h * 0.1)
            forehead_h = int(h * 0.3)
            forehead = opencv_img[forehead_y:forehead_y + forehead_h, x:x + w]

            if forehead.size == 0:
                return "normal"

            # Get average green channel intensity (PPG signal)
            green_channel = forehead[:, :, 1]  # Green channel
            avg_green = np.mean(green_channel)

            # Add to buffer
            self.heart_rate_buffer.append(avg_green)
            if len(self.heart_rate_buffer) > 300:  # ~30 seconds at 10fps for better analysis
                self.heart_rate_buffer.pop(0)

            # Calculate heart rate using Eulerian Video Magnification approach
            if len(self.heart_rate_buffer) >= 150:
                signal = np.array(self.heart_rate_buffer, dtype=np.float64)
                signal = signal - np.mean(signal)  # DC removal

                # Apply Eulerian magnification: bandpass filter for heart rate frequencies (0.5-3.5 Hz)
                # Assuming 10 fps sampling rate
                fs = 10.0  # Sampling frequency
                lowcut = 0.5  # 30 BPM
                highcut = 3.5  # 210 BPM

                # Design bandpass filter
                nyquist = fs / 2
                low = lowcut / nyquist
                high = highcut / nyquist
                b, a = signal.butter(4, [low, high], btype='band')

                # Apply filter
                filtered_signal = signal.filtfilt(b, a, signal)

                # Amplify the filtered signal (Eulerian magnification)
                alpha = 50  # Amplification factor
                magnified_signal = signal + alpha * filtered_signal

                # Find peaks in magnified signal
                peaks = []
                threshold = np.mean(magnified_signal) + 2 * np.std(magnified_signal)
                for i in range(1, len(magnified_signal) - 1):
                    if (magnified_signal[i] > magnified_signal[i-1] and
                        magnified_signal[i] > magnified_signal[i+1] and
                        magnified_signal[i] > threshold):
                        peaks.append(i)

                if len(peaks) >= 2:
                    # Estimate BPM
                    intervals = np.diff(peaks) / fs  # seconds between peaks
                    avg_interval = np.mean(intervals)
                    bpm = 60.0 / avg_interval

                    if bpm < 50:
                        return "low (possibly resting or meditating)"
                    elif bpm > 120:
                        return "elevated (possibly stressed or exercising)"
                    else:
                        return "normal"

            return "normal"

        except Exception as e:
            return f"Heart rate analysis error: {e}"

    def _analyze_pupil_dilation(self, image: Image.Image) -> str:
        """Analyze pupil dilation for emotional state"""
        try:
            opencv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2GRAY)

            # Detect faces
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

            if len(faces) == 0:
                return "normal"

            # Use first face
            (x, y, w, h) = faces[0]

            # Detect eyes in face region
            face_roi = gray[y:y+h, x:x+w]
            eyes = self.eye_cascade.detectMultiScale(face_roi, 1.1, 3)

            if len(eyes) == 0:
                return "normal"

            # Analyze each eye for pupil size
            pupil_sizes = []
            for (ex, ey, ew, eh) in eyes[:2]:  # Max 2 eyes
                eye_roi = face_roi[ey:ey+eh, ex:ex+ew]

                # Threshold to find dark regions (pupils)
                _, thresh = cv2.threshold(eye_roi, 30, 255, cv2.THRESH_BINARY_INV)

                # Find contours
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if contours:
                    # Largest contour is likely the pupil
                    largest_contour = max(contours, key=cv2.contourArea)
                    pupil_area = cv2.contourArea(largest_contour)
                    pupil_sizes.append(pupil_area)

            if pupil_sizes:
                avg_pupil_size = np.mean(pupil_sizes)

                # Baseline pupil size (this would need calibration)
                # For now, use relative assessment
                if avg_pupil_size > 200:  # Dilated
                    return "dilated (possibly interested or stressed)"
                elif avg_pupil_size < 50:  # Constricted
                    return "constricted (possibly bored or focused)"
                else:
                    return "normal"

            return "normal"

        except Exception as e:
            return f"Pupil dilation analysis error: {e}"

    def _analyze_saccades(self, image: Image.Image) -> str:
        """Analyze saccadic eye movements for attention and focus"""
        try:
            opencv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2GRAY)

            # Detect faces
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

            if len(faces) == 0:
                return "normal"

            # Use first face
            (x, y, w, h) = faces[0]

            # Detect eyes in face region
            face_roi = gray[y:y+h, x:x+w]
            eyes = self.eye_cascade.detectMultiScale(face_roi, 1.1, 3)

            if len(eyes) == 0:
                return "normal"

            # Track eye center positions
            current_eye_centers = []
            for (ex, ey, ew, eh) in eyes[:2]:  # Max 2 eyes
                eye_center_x = x + ex + ew // 2
                eye_center_y = y + ey + eh // 2
                current_eye_centers.append((eye_center_x, eye_center_y))

            # Add to history
            self.eye_positions.append(current_eye_centers)
            if len(self.eye_positions) > 50:  # Keep last 50 frames
                self.eye_positions.pop(0)

            # Analyze saccades if we have enough history
            if len(self.eye_positions) >= 10:
                # Calculate velocity for each eye
                saccade_detected = False
                for eye_idx in range(min(len(current_eye_centers), 2)):  # Max 2 eyes
                    if len(self.eye_positions) >= 2:
                        # Calculate velocity over last few frames
                        velocities = []
                        for i in range(max(1, len(self.eye_positions) - 5), len(self.eye_positions)):
                            if i > 0 and eye_idx < len(self.eye_positions[i]) and eye_idx < len(self.eye_positions[i-1]):
                                prev_pos = self.eye_positions[i-1][eye_idx]
                                curr_pos = self.eye_positions[i][eye_idx]
                                velocity = np.sqrt((curr_pos[0] - prev_pos[0])**2 + (curr_pos[1] - prev_pos[1])**2)
                                velocities.append(velocity)

                        if velocities:
                            avg_velocity = np.mean(velocities)
                            max_velocity = np.max(velocities)

                            # Saccade threshold: rapid movement > 50 pixels/frame
                            if max_velocity > 50:
                                saccade_detected = True
                                break

                if saccade_detected:
                    return "detected (rapid scanning, possibly searching or reading)"
                else:
                    # Check for fixation (low velocity, focused attention)
                    if len(self.eye_positions) >= 5:
                        recent_positions = self.eye_positions[-5:]
                        stable = True
                        for eye_idx in range(min(len(current_eye_centers), 2)):
                            if len(recent_positions[0]) > eye_idx:
                                base_pos = recent_positions[0][eye_idx]
                                for pos_list in recent_positions[1:]:
                                    if len(pos_list) > eye_idx:
                                        dist = np.sqrt((pos_list[eye_idx][0] - base_pos[0])**2 +
                                                     (pos_list[eye_idx][1] - base_pos[1])**2)
                                        if dist > 20:  # Movement threshold
                                            stable = False
                                            break
                                if not stable:
                                    break

                        if stable:
                            return "fixation (focused attention)"
                        else:
                            return "smooth pursuit (following moving object)"

            return "normal"

        except Exception as e:
            return f"Saccade analysis error: {e}"

    def _detect_objects(self, image: Image.Image):
        """Detect and track objects using YOLOv8"""
        if not self.yolo_model:
            return

        try:
            # Convert PIL to numpy
            img_array = np.array(image)

            # Run YOLO detection
            results = self.yolo_model(img_array)

            # Clear previous positions
            self.object_positions = {}

            # Process results
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get class name
                        cls = int(box.cls[0])
                        name = result.names[cls]

                        if name in self.tracked_objects:
                            # Get bbox coordinates
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)

                            self.object_positions[name] = (x, y, w, h)

        except Exception as e:
            print(f"Object detection error: {e}")

    def find_object_location(self, object_name: str) -> str:
        """Find and highlight the location of a physical object"""
        from jason.core.overlay import play_chirp

        if object_name.lower() in self.object_positions:
            x, y, w, h = self.object_positions[object_name.lower()]

            # Highlight with red ring on overlay
            # Since overlay is for screen, assume webcam is displayed on screen
            # Draw a red circle around the object
            rectangles = [{
                "x": x - 10,
                "y": y - 10,
                "w": w + 20,
                "h": h + 20,
                "label": f"{object_name.upper()} LOCATED"
            }]

            # Show overlay with red ring
            from jason.core.overlay import highlight_screen_areas
            highlight_screen_areas(rectangles)

            # Play chirp for located object
            play_chirp("spatial")

            return f"{object_name.capitalize()} located at position ({x}, {y}) with size {w}x{h}. Highlighting on screen."
        else:
            return f"{object_name.capitalize()} not detected in current view."

    def monitor_heart_rate(self) -> str:
        """Monitor current heart rate status"""
        img = self._capture_webcam_image()
        return f"Heart rate status: {self._analyze_heart_rate(img)}"

    def monitor_pupil_dilation(self) -> str:
        """Monitor current pupil dilation status"""
        img = self._capture_webcam_image()
        return f"Pupil dilation status: {self._analyze_pupil_dilation(img)}"

    def monitor_saccades(self) -> str:
        """Monitor current saccade status"""
        img = self._capture_webcam_image()
        return f"Saccade status: {self._analyze_saccades(img)}"

    def sketch_to_openscad(self, output_path: str = "design.scad") -> str:
        """Convert a hand-drawn sketch to OpenSCAD code for 3D printing"""
        try:
            # Capture current webcam/screen for the sketch
            img = self._capture_webcam_image()

            # Use Gemini to analyze the sketch and generate OpenSCAD code
            prompt = """
            Analyze this hand-drawn sketch and generate OpenSCAD code to create a 3D printable version.

            Look at the shapes, dimensions, and features in the sketch. Convert them to OpenSCAD primitives like:
            - cube([x,y,z]) for rectangular shapes
            - cylinder(h,r) for cylindrical shapes
            - sphere(r) for spherical shapes
            - translate([x,y,z]) for positioning
            - rotate([x,y,z]) for rotation
            - union() and difference() for combining shapes

            Generate complete, valid OpenSCAD code that can be compiled and 3D printed.
            Include proper dimensions (assume sketch is drawn on standard paper size).
            Make the design printable with appropriate wall thickness and supports.

            Return only the OpenSCAD code, no explanations.
            """

            openscad_code = self._analyze_image(img, prompt)

            if openscad_code and "```" in openscad_code:
                # Extract code from markdown if present
                code_start = openscad_code.find("```openscad") or openscad_code.find("```")
                code_end = openscad_code.find("```", code_start + 3)
                if code_start != -1 and code_end != -1:
                    openscad_code = openscad_code[code_start:code_end].replace("```openscad", "").replace("```", "").strip()

            # Save to file
            with open(output_path, 'w') as f:
                f.write(openscad_code)

            return f"OpenSCAD code generated and saved to {output_path}. Code preview:\n{openscad_code[:200]}..."

        except Exception as e:
            return f"Sketch to OpenSCAD conversion failed: {e}"
