"""
J.A.S.O.N. Tactical Briefing - Audio & Voice Module
Biometric Voice-Lock: Voiceprint Authentication System
"""

import pyttsx3
try:
    import speech_recognition as sr
    speech_recognition_available = True
except ImportError:
    speech_recognition_available = False

# Add Piper TTS support (Authorized: Zero-cost local synthesis)
piper_available = False
try:
    import subprocess
    # Check if piper command is available
    result = subprocess.run(['which', 'piper'], capture_output=True, text=True)
    piper_available = result.returncode == 0
except:
    piper_available = False

from langchain_core.tools import Tool
from typing import Optional, Dict, Any, List
# Add voiceprint verification
librosa_available = False
try:
    import librosa
    librosa_available = True
except ImportError:
    pass

speechbrain_available = False
try:
    import speechbrain
    from speechbrain.inference import SpeakerRecognition
    speechbrain_available = True
except ImportError:
    pass

import numpy as np
from typing import Optional, Dict, Any, List
import os
import json
import hashlib
from datetime import datetime
import threading
import time

class AudioManager:
    def __init__(self, elevenlabs_api_key: str = ""):
        self.elevenlabs_api_key = elevenlabs_api_key
        # Initialize TTS engines
        self.pyttsx_engine = pyttsx3.init()

        # Piper TTS (Authorized: Zero-cost local synthesis)
        self.piper_available = piper_available
        self.piper_model_path = "/usr/share/piper/en_US-lessac-medium.onnx"  # Default Piper model path

        # Speech recognition (Authorized: Gemini API integration planned)
        self.recognizer = None
        if speech_recognition_available:
            self.recognizer = sr.Recognizer()

        # Biometric voice-lock settings
        self.voice_lock_enabled = True  # Enable by default for security
        self.voiceprint_database = self._load_voiceprint_database()
        self.current_user_verified = False
        self.verification_timeout = 300  # 5 minutes verification timeout
        self.last_verification_time = 0

        # Voiceprint verification settings
        self.verification_threshold = 0.8  # Similarity threshold for authentication
        self.min_enrollment_samples = 3  # Minimum voice samples for enrollment

        # Speaker recognition model
        self.speaker_recognition = None
        if speechbrain_available:
            try:
                self.speaker_recognition = SpeakerRecognition.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb",
                    savedir="tmp/speechbrain"
                )
            except Exception as e:
                print(f"SpeechBrain speaker recognition failed: {e}")

        # Voice analysis for authentication
        self.voice_features_cache = {}

        # Create tools
        self.listen_tool = Tool(
            name="Listen",
            func=self.listen,
            description="Listen for voice input and process with biometric verification"
        )

        self.speak_tool = Tool(
            name="Speak",
            func=self.speak,
            description="Convert text to speech output"
        )

        self.voice_lock_tool = Tool(
            name="Voice Lock",
            func=self.voice_lock_status,
            description="Check biometric voice-lock status"
        )

    def _load_voiceprint_database(self) -> Dict[str, Any]:
        """Load voiceprint database from file"""
        db_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'voiceprints.json')
        try:
            if os.path.exists(db_path):
                with open(db_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Failed to load voiceprint database: {e}")
        return {}

    def _save_voiceprint_database(self):
        """Save voiceprint database to file"""
        db_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'voiceprints.json')
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        try:
            with open(db_path, 'w') as f:
                json.dump(self.voiceprint_database, f, indent=2)
        except Exception as e:
            print(f"Failed to save voiceprint database: {e}")

    def enroll_voiceprint(self, user_id: str, audio_samples: List[np.ndarray]) -> Dict[str, Any]:
        """Enroll a user's voiceprint from multiple audio samples"""
        if not librosa_available:
            return {'success': False, 'error': 'Librosa not available for voiceprint analysis'}

        if len(audio_samples) < self.min_enrollment_samples:
            return {
                'success': False,
                'error': f'Insufficient samples. Need at least {self.min_enrollment_samples} voice samples'
            }

        try:
            # Extract features from all samples
            all_features = []
            for audio in audio_samples:
                features = self._extract_voice_features(audio)
                if features:
                    all_features.append(features)

            if not all_features:
                return {'success': False, 'error': 'Failed to extract features from audio samples'}

            # Average features across samples for robust voiceprint
            voiceprint = {}
            for feature_name in all_features[0].keys():
                values = [f[feature_name] for f in all_features if feature_name in f]
                if values:
                    voiceprint[feature_name] = float(np.mean(values))

            # Store voiceprint
            enrollment_data = {
                'voiceprint': voiceprint,
                'enrolled_at': datetime.now().isoformat(),
                'sample_count': len(audio_samples),
                'feature_count': len(voiceprint)
            }

            self.voiceprint_database[user_id] = enrollment_data
            self._save_voiceprint_database()

            return {
                'success': True,
                'user_id': user_id,
                'message': f'Voiceprint enrolled for {user_id} with {len(voiceprint)} features'
            }

        except Exception as e:
            return {'success': False, 'error': f'Enrollment failed: {str(e)}'}

    def verify_voiceprint(self, audio: np.ndarray, user_id: str = None) -> Dict[str, Any]:
        """Verify voice against enrolled voiceprints"""
        if not librosa_available:
            return {'success': False, 'error': 'Librosa not available for voice verification'}

        if not self.voiceprint_database:
            return {'success': False, 'error': 'No enrolled voiceprints found'}

        try:
            # Extract features from input audio
            input_features = self._extract_voice_features(audio)
            if not input_features:
                return {'success': False, 'error': 'Failed to extract features from audio'}

            best_match = None
            best_similarity = 0.0
            matched_user = None

            # Compare against all enrolled voiceprints
            for enrolled_user, data in self.voiceprint_database.items():
                if user_id and enrolled_user != user_id:
                    continue

                enrolled_features = data.get('voiceprint', {})

                # Calculate similarity
                similarity = self._calculate_voice_similarity(input_features, enrolled_features)

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = data
                    matched_user = enrolled_user

            # Check if similarity meets threshold
            if best_similarity >= self.verification_threshold:
                # Update verification status
                self.current_user_verified = True
                self.last_verification_time = time.time()

                return {
                    'success': True,
                    'verified_user': matched_user,
                    'similarity': best_similarity,
                    'confidence': (best_similarity - self.verification_threshold) / (1 - self.verification_threshold),
                    'message': f'Voice verified as {matched_user}'
                }
            else:
                return {
                    'success': False,
                    'error': 'Voice verification failed',
                    'best_similarity': best_similarity,
                    'threshold': self.verification_threshold,
                    'message': f'Voice not recognized. Similarity: {best_similarity:.2%}'
                }

        except Exception as e:
            return {'success': False, 'error': f'Verification failed: {str(e)}'}

    def _extract_voice_features(self, audio: np.ndarray, sample_rate: int = 16000) -> Optional[Dict[str, float]]:
        """Extract voice features for biometric analysis"""
        if not librosa_available or len(audio) == 0:
            return None

        try:
            # Ensure audio is mono
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)

            # Extract MFCCs (Mel-frequency cepstral coefficients)
            mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
            mfcc_means = np.mean(mfccs, axis=1)

            # Extract chroma features
            chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
            chroma_means = np.mean(chroma, axis=1)

            # Extract spectral features
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sample_rate))
            spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sample_rate))
            zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(audio))

            # Extract pitch features
            pitches, voiced_flag = librosa.core.pyin(audio, fmin=75, fmax=300, sr=sample_rate)
            valid_pitches = pitches[voiced_flag]
            pitch_mean = np.mean(valid_pitches) if len(valid_pitches) > 0 else 0
            pitch_std = np.std(valid_pitches) if len(valid_pitches) > 0 else 0

            # Compile features
            features = {}

            # MFCC features
            for i, mfcc_val in enumerate(mfcc_means):
                features[f'mfcc_{i}'] = float(mfcc_val)

            # Chroma features
            for i, chroma_val in enumerate(chroma_means):
                features[f'chroma_{i}'] = float(chroma_val)

            # Spectral features
            features['spectral_centroid'] = float(spectral_centroid)
            features['spectral_rolloff'] = float(spectral_rolloff)
            features['zero_crossing_rate'] = float(zero_crossing_rate)

            # Pitch features
            features['pitch_mean'] = float(pitch_mean)
            features['pitch_std'] = float(pitch_std)

            return features

        except Exception as e:
            print(f"Voice feature extraction failed: {e}")
            return None

    def _calculate_voice_similarity(self, features1: Dict[str, float], features2: Dict[str, float]) -> float:
        """Calculate similarity between two voice feature sets"""
        if not features1 or not features2:
            return 0.0

        # Find common features
        common_features = set(features1.keys()) & set(features2.keys())

        if not common_features:
            return 0.0

        # Calculate cosine similarity for each feature set
        similarities = []

        for feature in common_features:
            val1 = features1[feature]
            val2 = features2[feature]

            # Cosine similarity for this feature
            if val1 != 0 or val2 != 0:
                similarity = (val1 * val2) / (abs(val1) * abs(val2) + 1e-10)  # Add small epsilon
                similarities.append(similarity)

        # Average similarity across all features
        if similarities:
            return float(np.mean(similarities))
        else:
            return 0.0

    def is_voice_verified(self) -> bool:
        """Check if current user is voice verified"""
        if not self.voice_lock_enabled:
            return True  # Voice lock disabled

        # Check if verification has timed out
        if time.time() - self.last_verification_time > self.verification_timeout:
            self.current_user_verified = False

        return self.current_user_verified

    def require_voice_verification(self) -> str:
        """Require voice verification before proceeding"""
        return "Biometric Voice-Lock: Voice verification required. Please speak a verification phrase."

    def voice_lock_status(self) -> str:
        """Get voice-lock status"""
        if not self.voice_lock_enabled:
            return "Voice-Lock: DISABLED"

        status = "Voice-Lock: ENABLED\n"
        status += f"Current User Verified: {'Yes' if self.is_voice_verified() else 'No'}\n"
        status += f"Enrolled Users: {len(self.voiceprint_database)}\n"

        if self.voiceprint_database:
            status += "Enrolled Users:\n"
            for user_id, data in self.voiceprint_database.items():
                enrolled_at = data.get('enrolled_at', 'Unknown')
                samples = data.get('sample_count', 0)
                status += f"  - {user_id} (enrolled: {enrolled_at}, samples: {samples})\n"

        status += f"Verification Threshold: {self.verification_threshold:.1%}\n"
        status += f"Verification Timeout: {self.verification_timeout} seconds"

        return status

    def enroll_current_user(self, user_id: str) -> str:
        """Enroll the current user's voiceprint"""
        if not speech_recognition_available:
            return "Voice enrollment requires speech recognition"

        enrollment_samples = []

        print(f"Voice Enrollment for {user_id}")
        print("Please speak the following phrases clearly:")

        phrases = [
            "J.A.S.O.N. activate biometric voice-lock",
            "This is my voice for authentication",
            "Verify my identity through voice analysis"
        ]

        for i, phrase in enumerate(phrases):
            print(f"\nPhrase {i+1}: '{phrase}'")
            print("Press Enter when ready to speak...")

            try:
                input()  # Wait for user input

                # Record audio
                with sr.Microphone() as source:
                    print("Listening...")
                    self.recognizer.adjust_for_ambient_noise(source, duration=1)
                    audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=5)

                    # Convert to numpy array
                    audio_data = np.frombuffer(audio.get_raw_data(), dtype=np.int16).astype(np.float32) / 32768.0
                    enrollment_samples.append(audio_data)

                    print(f"Sample {i+1} recorded")

            except Exception as e:
                print(f"Failed to record sample {i+1}: {e}")
                continue

        if len(enrollment_samples) >= self.min_enrollment_samples:
            result = self.enroll_voiceprint(user_id, enrollment_samples)

            if result['success']:
                return f"Voice enrollment successful for {user_id}"
            else:
                return f"Voice enrollment failed: {result.get('error', 'Unknown error')}"
        else:
            return f"Voice enrollment failed: Only {len(enrollment_samples)} samples recorded (need {self.min_enrollment_samples})"

    def listen(self, timeout: int = 5) -> str:
        """Listen for voice input with biometric verification"""
        if not speech_recognition_available:
            return "Speech recognition not available"

        try:
            with sr.Microphone() as source:
                print("Listening...")
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)

                # Listen for audio
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=10)

                # Convert to text
                try:
                    text = self.recognizer.recognize_google(audio)
                    print(f"Heard: {text}")

                    # Perform voice verification if enabled
                    if self.voice_lock_enabled:
                        # Convert audio to numpy for verification
                        audio_data = np.frombuffer(audio.get_raw_data(), dtype=np.int16).astype(np.float32) / 32768.0

                        verification_result = self.verify_voiceprint(audio_data)

                        if verification_result['success']:
                            print(f"Voice verified: {verification_result['verified_user']}")
                            return f"Heard: {text}"
                        else:
                            print(f"Voice verification failed: {verification_result.get('message', 'Unknown')}")
                            return f"Voice verification failed. Please authenticate with your enrolled voice."

                    return f"Heard: {text}"

                except sr.UnknownValueError:
                    return "Could not understand audio"
                except sr.RequestError as e:
                    return f"Speech recognition error: {e}"

        except Exception as e:
            return f"Listening error: {e}"

    def speak(self, text: str) -> str:
        """Convert text to speech using Piper TTS (zero-cost local synthesis)"""
        try:
            # Priority: Piper (zero-API) > pyttsx3 (fallback)
            if self.piper_available:
                return self._piper_speak(text)
            else:
                # Use pyttsx3 as fallback
                self.pyttsx_engine.say(text)
                self.pyttsx_engine.runAndWait()
                return f"Tactical update: {text[:50]}..."

        except Exception as e:
            return f"Voice synthesis failed: {e}"

    def set_voice_properties(self, rate: int = 200, volume: float = 1.0):
        """Configure voice properties"""
        self.pyttsx_engine.setProperty('rate', rate)
        self.pyttsx_engine.setProperty('volume', volume)

    def get_voice_info(self) -> str:
        """Get current voice configuration"""
        rate = self.pyttsx_engine.getProperty('rate')
        volume = self.pyttsx_engine.getProperty('volume')
        voice = self.pyttsx_engine.getProperty('voice')

        return f"Voice: {voice}, Rate: {rate}, Volume: {volume}"
