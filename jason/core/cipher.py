"""
J.A.S.O.N. Cipher Protocol
Social Engineering & Truth Engine: Real-time Voice & Facial Analysis during Calls
"""

import os
import time
import threading
import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from datetime import datetime
import json

# Audio processing
try:
    import librosa
    librosa_available = True
except ImportError:
    librosa_available = False

try:
    import speech_recognition as sr
    speech_recognition_available = True
except ImportError:
    speech_recognition_available = False

# Machine learning for deception detection
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    sklearn_available = True
except ImportError:
    sklearn_available = False

logger = logging.getLogger(__name__)

class CipherManager:
    """Cipher Protocol: Social Engineering & Truth Engine with real-time voice & facial analysis"""

    def __init__(self, config: dict = None):
        self.config = config or {}

        # Cipher settings
        cipher_config = self.config.get('cipher', {})
        self.analysis_active = False
        self.call_recording_active = False

        # Voice analysis settings
        self.voice_sample_rate = cipher_config.get('sample_rate', 16000)
        self.voice_analysis_window = cipher_config.get('analysis_window_seconds', 5)

        # Deception detection thresholds
        self.deception_threshold = cipher_config.get('deception_threshold', 0.7)
        self.stress_threshold = cipher_config.get('stress_threshold', 0.6)

        # Voice feature extraction
        self.voice_features = {
            'pitch_variation': [],
            'speech_rate': [],
            'pause_frequency': [],
            'volume_variation': [],
            'tone_stability': []
        }

        # Facial analysis integration
        self.facial_baseline = {}
        self.emotion_history = []

        # Deception detection model
        self.deception_model = None
        self.feature_scaler = None
        self._initialize_deception_model()

        # Call recording
        self.current_call_audio = []
        self.call_transcript = []
        self.call_analysis = {}

        # Analysis results
        self.analysis_results = {
            'deception_probability': 0.0,
            'stress_level': 0.0,
            'confidence_level': 0.0,
            'analysis_summary': '',
            'recommendations': []
        }

        # Data storage
        self.data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'cipher')
        os.makedirs(self.data_dir, exist_ok=True)

    def _initialize_deception_model(self):
        """Initialize machine learning model for deception detection"""
        if not sklearn_available:
            logger.warning("scikit-learn not available - deception detection limited")
            return

        # Create a simple deception detection model
        # In production, this would be trained on real data
        self.deception_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.feature_scaler = StandardScaler()

        # Initialize with baseline features (would be trained on real data)
        baseline_features = np.random.rand(100, 10)  # 100 samples, 10 features
        baseline_labels = np.random.randint(0, 2, 100)  # Binary deception labels

        # Fit the model
        try:
            scaled_features = self.feature_scaler.fit_transform(baseline_features)
            self.deception_model.fit(scaled_features, baseline_labels)
        except Exception as e:
            logger.warning(f"Failed to initialize deception model: {e}")

    def start_call_analysis(self, participant_name: str = "Unknown") -> bool:
        """Start real-time call analysis during a meeting/call"""
        if self.analysis_active:
            return False

        try:
            self.analysis_active = True
            self.call_recording_active = True
            self.current_call_audio = []
            self.call_transcript = []
            self.call_analysis = {
                'participant': participant_name,
                'start_time': datetime.now().isoformat(),
                'voice_features': {},
                'facial_analysis': {},
                'deception_indicators': [],
                'stress_indicators': [],
                'transcript': []
            }

            logger.info(f"Cipher Protocol: Call analysis started for {participant_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to start call analysis: {e}")
            self.analysis_active = False
            return False

    def stop_call_analysis(self) -> Dict[str, Any]:
        """Stop call analysis and return final results"""
        if not self.analysis_active:
            return {'error': 'Call analysis not active'}

        self.analysis_active = False
        self.call_recording_active = False

        # Final analysis
        final_results = self._generate_final_analysis()

        # Save analysis
        self._save_call_analysis(final_results)

        logger.info("Cipher Protocol: Call analysis completed")
        return final_results

    def analyze_voice_segment(self, audio_data: np.ndarray, sample_rate: int = 16000) -> Dict[str, Any]:
        """Analyze a segment of voice audio for deception indicators"""
        if not librosa_available:
            return {'error': 'Librosa not available for voice analysis'}

        try:
            voice_results = {
                'timestamp': datetime.now().isoformat(),
                'features': {},
                'deception_probability': 0.0,
                'stress_level': 0.0
            }

            # Extract voice features
            features = self._extract_voice_features(audio_data, sample_rate)
            voice_results['features'] = features

            # Analyze for deception
            deception_analysis = self._analyze_voice_deception(features)
            voice_results.update(deception_analysis)

            # Update rolling analysis
            self._update_voice_history(features)

            return voice_results

        except Exception as e:
            logger.error(f"Voice segment analysis failed: {e}")
            return {'error': str(e)}

    def analyze_facial_segment(self, face_image: np.ndarray, face_landmarks: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Analyze facial expressions and micro-expressions during call"""
        try:
            facial_results = {
                'timestamp': datetime.now().isoformat(),
                'emotions': {},
                'micro_expressions': [],
                'eye_contact': 0.0,
                'facial_tension': 0.0
            }

            # Analyze facial expressions
            if hasattr(self, '_analyze_facial_expressions'):
                emotions = self._analyze_facial_expressions(face_image, face_landmarks)
                facial_results['emotions'] = emotions

            # Detect micro-expressions
            micro_expressions = self._detect_micro_expressions(face_image)
            facial_results['micro_expressions'] = micro_expressions

            # Analyze eye contact
            eye_contact = self._analyze_eye_contact(face_landmarks)
            facial_results['eye_contact'] = eye_contact

            # Measure facial tension
            tension = self._measure_facial_tension(face_landmarks)
            facial_results['facial_tension'] = tension

            # Update emotion history
            self.emotion_history.append({
                'timestamp': facial_results['timestamp'],
                'emotions': facial_results['emotions'],
                'tension': tension,
                'eye_contact': eye_contact
            })

            # Limit history
            if len(self.emotion_history) > 100:
                self.emotion_history.pop(0)

            return facial_results

        except Exception as e:
            logger.error(f"Facial segment analysis failed: {e}")
            return {'error': str(e)}

    def _extract_voice_features(self, audio: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Extract acoustic features from voice audio"""
        features = {}

        try:
            # Pitch analysis
            pitches, voiced_flag, _ = librosa.pyin(audio, fmin=75, fmax=600, sr=sample_rate)
            valid_pitches = pitches[voiced_flag]
            if len(valid_pitches) > 0:
                features['pitch_mean'] = float(np.mean(valid_pitches))
                features['pitch_std'] = float(np.std(valid_pitches))
                features['pitch_range'] = float(np.max(valid_pitches) - np.min(valid_pitches))
            else:
                features['pitch_mean'] = 0.0
                features['pitch_std'] = 0.0
                features['pitch_range'] = 0.0

            # Speech rate (syllables per second)
            # Simple approximation based on zero crossings
            zero_crossings = librosa.zero_crossings(audio)
            features['speech_rate'] = float(np.sum(zero_crossings) / len(audio) * sample_rate / 100)

            # Volume analysis
            rms = librosa.feature.rms(y=audio)
            features['volume_mean'] = float(np.mean(rms))
            features['volume_std'] = float(np.std(rms))

            # Pause detection (simple silence detection)
            silence_threshold = np.mean(np.abs(audio)) * 0.1
            silence_frames = np.sum(np.abs(audio) < silence_threshold)
            features['pause_ratio'] = float(silence_frames / len(audio))

            # Spectral features
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)
            features['spectral_centroid_mean'] = float(np.mean(spectral_centroid))

            # MFCCs for voice quality
            mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
            for i in range(13):
                features[f'mfcc_{i}'] = float(np.mean(mfccs[i]))

        except Exception as e:
            logger.warning(f"Voice feature extraction failed: {e}")
            # Provide default values
            features = {k: 0.0 for k in ['pitch_mean', 'pitch_std', 'pitch_range', 'speech_rate',
                                       'volume_mean', 'volume_std', 'pause_ratio', 'spectral_centroid_mean'] +
                       [f'mfcc_{i}' for i in range(13)]}

        return features

    def _analyze_voice_deception(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze voice features for deception indicators"""
        deception_indicators = []

        # Rule-based deception detection
        deception_score = 0.0

        # High pitch variation (stress/nervousness)
        if features.get('pitch_std', 0) > 50:
            deception_score += 0.2
            deception_indicators.append('high_pitch_variation')

        # Irregular speech rate
        speech_rate = features.get('speech_rate', 0)
        if speech_rate > 5 or speech_rate < 1:
            deception_score += 0.15
            deception_indicators.append('irregular_speech_rate')

        # Excessive pauses
        if features.get('pause_ratio', 0) > 0.3:
            deception_score += 0.2
            deception_indicators.append('excessive_pauses')

        # Volume inconsistency
        if features.get('volume_std', 0) > 0.1:
            deception_score += 0.15
            deception_indicators.append('volume_inconsistency')

        # Machine learning prediction (if available)
        if self.deception_model and sklearn_available:
            try:
                feature_vector = np.array([list(features.values())])
                scaled_features = self.feature_scaler.transform(feature_vector)
                ml_prediction = self.deception_model.predict_proba(scaled_features)[0][1]
                deception_score = (deception_score + ml_prediction) / 2  # Blend rule-based and ML
            except Exception as e:
                logger.warning(f"ML deception prediction failed: {e}")

        # Stress analysis
        stress_indicators = []
        stress_score = 0.0

        if features.get('pitch_std', 0) > 40:
            stress_score += 0.3
            stress_indicators.append('elevated_pitch_variation')

        if features.get('speech_rate', 0) > 4:
            stress_score += 0.2
            stress_indicators.append('rapid_speech')

        if features.get('pause_ratio', 0) > 0.4:
            stress_score += 0.25
            stress_indicators.append('frequent_pauses')

        return {
            'deception_probability': min(deception_score, 1.0),
            'deception_indicators': deception_indicators,
            'stress_level': min(stress_score, 1.0),
            'stress_indicators': stress_indicators
        }

    def _analyze_facial_expressions(self, face_image: np.ndarray, landmarks: Optional[np.ndarray]) -> Dict[str, Any]:
        """Analyze facial expressions for emotions"""
        # This would use a facial expression recognition model
        # For now, return placeholder analysis
        emotions = {
            'happy': 0.1,
            'sad': 0.05,
            'angry': 0.1,
            'fear': 0.1,
            'surprise': 0.05,
            'neutral': 0.6
        }

        # In production, this would use:
        # - Facial landmark analysis
        # - Deep learning models (FER, AffectNet)
        # - Micro-expression detection

        return emotions

    def _detect_micro_expressions(self, face_image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect micro-expressions that may indicate deception"""
        # Micro-expression detection is complex and requires specialized models
        # This is a placeholder for the concept

        micro_expressions = []

        # Simulate occasional micro-expression detection
        import random
        if random.random() < 0.1:  # 10% chance
            expressions = ['brief_frown', 'eyebrow_raise', 'lip_compression', 'eye_widening']
            micro_expressions.append({
                'type': random.choice(expressions),
                'intensity': random.uniform(0.3, 0.8),
                'duration_ms': random.randint(100, 500),
                'timestamp': datetime.now().isoformat()
            })

        return micro_expressions

    def _analyze_eye_contact(self, landmarks: Optional[np.ndarray]) -> float:
        """Analyze eye contact based on gaze direction"""
        # This would use eye landmark analysis
        # Placeholder implementation
        return 0.7  # 70% eye contact

    def _measure_facial_tension(self, landmarks: Optional[np.ndarray]) -> float:
        """Measure facial muscle tension"""
        # This would analyze landmark distances and angles
        # Placeholder implementation
        return 0.3  # Low tension

    def _update_voice_history(self, features: Dict[str, Any]):
        """Update voice feature history for trend analysis"""
        for feature_name, value in features.items():
            if feature_name not in self.voice_features:
                self.voice_features[feature_name] = []
            self.voice_features[feature_name].append(value)

            # Keep only recent history
            if len(self.voice_features[feature_name]) > 50:
                self.voice_features[feature_name].pop(0)

    def _generate_final_analysis(self) -> Dict[str, Any]:
        """Generate final call analysis summary"""
        final_analysis = {
            'participant': self.call_analysis.get('participant', 'Unknown'),
            'duration_minutes': (datetime.now() - datetime.fromisoformat(self.call_analysis['start_time'])).total_seconds() / 60,
            'overall_deception_probability': 0.0,
            'overall_stress_level': 0.0,
            'key_findings': [],
            'recommendations': [],
            'confidence_level': 0.0
        }

        # Analyze voice trends
        if self.voice_features:
            deception_probs = []
            stress_levels = []

            # Calculate average deception and stress over the call
            for features in zip(*self.voice_features.values()):
                if len(features) >= 10:  # Need minimum features for analysis
                    feature_dict = dict(zip(self.voice_features.keys(), features))
                    analysis = self._analyze_voice_deception(feature_dict)
                    deception_probs.append(analysis['deception_probability'])
                    stress_levels.append(analysis['stress_level'])

            if deception_probs:
                final_analysis['overall_deception_probability'] = float(np.mean(deception_probs))
                final_analysis['overall_stress_level'] = float(np.mean(stress_levels))

        # Generate key findings
        if final_analysis['overall_deception_probability'] > self.deception_threshold:
            final_analysis['key_findings'].append("High deception probability detected")
            final_analysis['recommendations'].append("Exercise caution with sensitive information")

        if final_analysis['overall_stress_level'] > self.stress_threshold:
            final_analysis['key_findings'].append("Elevated stress levels observed")
            final_analysis['recommendations'].append("Consider rescheduling or providing breaks")

        # Confidence assessment
        final_analysis['confidence_level'] = min(0.8, len(deception_probs) / 50)  # Higher confidence with more data

        return final_analysis

    def _save_call_analysis(self, analysis: Dict[str, Any]):
        """Save call analysis to file"""
        filename = f"call_analysis_{int(time.time())}.json"
        filepath = os.path.join(self.data_dir, filename)

        try:
            with open(filepath, 'w') as f:
                json.dump({
                    'analysis': analysis,
                    'call_data': self.call_analysis,
                    'raw_features': self.voice_features
                }, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save call analysis: {e}")

    def get_current_analysis(self) -> Dict[str, Any]:
        """Get current real-time analysis status"""
        return {
            'analysis_active': self.analysis_active,
            'current_participant': self.call_analysis.get('participant', 'None'),
            'deception_probability': self.analysis_results.get('deception_probability', 0.0),
            'stress_level': self.analysis_results.get('stress_level', 0.0),
            'confidence_level': self.analysis_results.get('confidence_level', 0.0),
            'recent_indicators': self.call_analysis.get('deception_indicators', [])[-3:]
        }

    def calibrate_baseline(self, calibration_audio: np.ndarray, calibration_faces: List[np.ndarray]):
        """Calibrate baseline behavior for more accurate deception detection"""
        logger.info("Calibrating Cipher Protocol baseline...")

        # This would analyze normal behavior to establish baselines
        # Placeholder implementation
        self.facial_baseline = {
            'average_emotions': {'neutral': 0.6, 'happy': 0.3, 'focused': 0.1},
            'baseline_tension': 0.2,
            'calibration_timestamp': datetime.now().isoformat()
        }

        logger.info("Cipher Protocol baseline calibration completed")

    def export_analysis_history(self, filepath: str) -> bool:
        """Export analysis history to file"""
        try:
            history_data = {
                'export_timestamp': datetime.now().isoformat(),
                'baseline_calibration': self.facial_baseline,
                'analysis_history': []  # Would include historical analyses
            }

            with open(filepath, 'w') as f:
                json.dump(history_data, f, indent=2, default=str)

            return True
        except Exception as e:
            logger.error(f"Failed to export analysis history: {e}")
            return False
