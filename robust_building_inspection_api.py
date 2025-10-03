# FINAL ROBUST BUILDING INSPECTION API
import tensorflow as tf
import numpy as np
from PIL import Image
import base64
import io
import json
import os
import cv2
import tempfile
from pathlib import Path

class RobustBuildingInspectionAPI:
    """
    Production-ready building inspection API
    Handles images, videos, and various file formats with robust error handling
    """
    
    def __init__(self, crack_model_path, moisture_model_path):
        print("🚀 Loading production inspection models...")
        try:
            self.crack_model = tf.keras.models.load_model(crack_model_path)
            self.moisture_model = tf.keras.models.load_model(moisture_model_path)
            self.models_loaded = True
            print("✅ Models loaded successfully!")
        except Exception as e:
            self.models_loaded = False
            print(f"❌ Model loading failed: {e}")
    
    def _preprocess_image(self, image):
        """Robust image preprocessing with format handling"""
        try:
            # Handle different image modes
            if image.mode in ('RGBA', 'LA', 'P'):
                image = image.convert('RGB')
            elif image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize and normalize
            image = image.resize((224, 224))
            image_array = np.array(image) / 255.0
            
            # Ensure 3 channels
            if len(image_array.shape) == 2:  # Grayscale
                image_array = np.stack([image_array] * 3, axis=-1)
            elif image_array.shape[2] == 4:  # RGBA
                image_array = image_array[:, :, :3]
            
            return np.expand_dims(image_array, axis=0)
            
        except Exception as e:
            raise ValueError(f"Image preprocessing failed: {e}")
    
    def _analyze_frame(self, image_array):
        """Analyze a single image frame"""
        if not self.models_loaded:
            return {"error": "Models not loaded"}
        
        try:
            crack_pred = self.crack_model.predict(image_array, verbose=0)[0][0]
            moisture_pred = self.moisture_model.predict(image_array, verbose=0)[0][0]
            
            return {
                "crack_detection": {
                    "detected": bool(crack_pred > 0.75),
                    "confidence": float(crack_pred),
                    "severity": self._get_severity(crack_pred)
                },
                "moisture_detection": {
                    "detected": bool(moisture_pred > 0.995),  # ← NEW THRESHOLD
                    "confidence": float(moisture_pred),
                    "severity": self._get_severity(moisture_pred)
                },
                "overall_assessment": {
                    "condition_score": float(1.0 - max(crack_pred, moisture_pred)),
                    "risk_level": self._get_risk_level(crack_pred, moisture_pred),
                    "urgent_action_required": bool(crack_pred > 0.9 or moisture_pred > 0.9)
                }
            }
        except Exception as e:
            return {"error": f"Prediction failed: {e}"}
    
    def predict_image(self, image_input):
        """Predict from image file, path, or bytes"""
        try:
            # Handle different input types
            if isinstance(image_input, (str, Path)):
                if not os.path.exists(image_input):
                    return {"error": f"File not found: {image_input}"}
                image = Image.open(image_input)
            elif isinstance(image_input, bytes):
                image = Image.open(io.BytesIO(image_input))
            elif isinstance(image_input, Image.Image):
                image = image_input
            else:
                return {"error": f"Unsupported input type: {type(image_input)}"}
            
            # Validate image size (max 50MB)
            if hasattr(image_input, 'tell'):  # File-like object
                image_input.seek(0, 2)  # Seek to end
                size = image_input.tell()
                image_input.seek(0)  # Seek back to start
                if size > 50 * 1024 * 1024:  # 50MB
                    return {"error": "Image too large (max 50MB)"}
            
            # Preprocess and analyze
            processed_image = self._preprocess_image(image)
            result = self._analyze_frame(processed_image)
            
            if "error" not in result:
                result["metadata"] = {
                    "timestamp": float(tf.timestamp().numpy()),
                    "model_version": "production_v2.0",
                    "input_type": "image",
                    "processing_time_ms": 0  # Would calculate actual time
                }
            
            return result
            
        except Exception as e:
            return {"error": f"Image processing failed: {e}"}
    
    def predict_video(self, video_path, max_duration=300, frame_interval=2):
        """
        Analyze video by extracting frames
        - max_duration: maximum video duration in seconds (5 minutes default)
        - frame_interval: analyze every Nth frame (2 = every 2nd frame)
        """
        try:
            if not os.path.exists(video_path):
                return {"error": f"Video file not found: {video_path}"}
            
            # Check file size (max 500MB)
            file_size = os.path.getsize(video_path) / (1024 * 1024)
            if file_size > 500:
                return {"error": f"Video too large: {file_size:.1f}MB (max 500MB)"}
            
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return {"error": "Could not open video file"}
            
            # Get video info
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            
            if duration > max_duration:
                return {"error": f"Video too long: {duration:.1f}s (max {max_duration}s)"}
            
            print(f"📹 Analyzing video: {total_frames} frames, {duration:.1f}s duration")
            
            frame_results = []
            frame_count = 0
            analyzed_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Analyze every Nth frame
                if frame_count % frame_interval == 0:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)
                    
                    # Analyze frame
                    result = self.predict_image(pil_image)
                    if "error" not in result:
                        result["frame_number"] = frame_count
                        result["timestamp_seconds"] = frame_count / fps
                        frame_results.append(result)
                        analyzed_count += 1
                
                frame_count += 1
            
            cap.release()
            
            if not frame_results:
                return {"error": "No frames could be analyzed"}
            
            # Aggregate results
            return self._aggregate_video_results(frame_results, duration, analyzed_count)
            
        except Exception as e:
            return {"error": f"Video processing failed: {e}"}
    
    def _aggregate_video_results(self, frame_results, duration, analyzed_count):
        """Aggregate results from multiple video frames"""
        crack_confidences = []
        moisture_confidences = []
        urgent_frames = []
        
        for i, result in enumerate(frame_results):
            crack_confidences.append(result["crack_detection"]["confidence"])
            moisture_confidences.append(result["moisture_detection"]["confidence"])
            
            if result["overall_assessment"]["urgent_action_required"]:
                urgent_frames.append(i)
        
        # Calculate overall metrics
        avg_crack = np.mean(crack_confidences)
        avg_moisture = np.mean(moisture_confidences)
        max_crack = np.max(crack_confidences)
        max_moisture = np.max(moisture_confidences)
        
        return {
            "video_analysis": {
                "total_frames_analyzed": analyzed_count,
                "video_duration_seconds": float(duration),
                "crack_detection_frames": sum(1 for r in frame_results if r["crack_detection"]["detected"]),
                "moisture_detection_frames": sum(1 for r in frame_results if r["moisture_detection"]["detected"]),
                "urgent_action_frames": len(urgent_frames),
                "confidence_metrics": {
                    "average_crack_confidence": float(avg_crack),
                    "average_moisture_confidence": float(avg_moisture),
                    "maximum_crack_confidence": float(max_crack),
                    "maximum_moisture_confidence": float(max_moisture)
                }
            },
            "frame_results": frame_results,
            "overall_assessment": {
                "condition_score": float(1.0 - max(max_crack, max_moisture)),
                "risk_level": self._get_risk_level(max_crack, max_moisture),
                "urgent_action_required": len(urgent_frames) > 0,
                "recommendation": self._get_video_recommendation(frame_results)
            },
            "metadata": {
                "timestamp": float(tf.timestamp().numpy()),
                "model_version": "production_v2.0",
                "input_type": "video",
                "processing_time_ms": 0
            }
        }
    
    def predict_from_base64(self, base64_string, input_type="image"):
        """Predict from base64 encoded image or video"""
        try:
            # Remove data URL prefix if present
            if ',' in base64_string:
                base64_string = base64_string.split(',')[1]
            
            file_data = base64.b64decode(base64_string)
            
            if input_type == "image":
                return self.predict_image(file_data)
            elif input_type == "video":
                # Save to temp file for video processing
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                    tmp_file.write(file_data)
                    tmp_path = tmp_file.name
                
                result = self.predict_video(tmp_path)
                os.unlink(tmp_path)  # Clean up temp file
                return result
            else:
                return {"error": f"Unsupported input type: {input_type}"}
                
        except Exception as e:
            return {"error": f"Base64 processing failed: {e}"}
    
    def _get_severity(self, confidence):
        if confidence < 0.5: return "none"
        elif confidence < 0.75: return "low"
        elif confidence < 0.9: return "medium"
        else: return "high"
    
    def _get_risk_level(self, crack_conf, moisture_conf):
        max_conf = max(crack_conf, moisture_conf)
        if max_conf < 0.5: return "low"
        elif max_conf < 0.75: return "moderate"
        elif max_conf < 0.9: return "high"
        else: return "critical"
    
    def _get_video_recommendation(self, frame_results):
        """Generate recommendation based on video analysis"""
        urgent_count = sum(1 for r in frame_results if r["overall_assessment"]["urgent_action_required"])
        crack_frames = sum(1 for r in frame_results if r["crack_detection"]["detected"])
        moisture_frames = sum(1 for r in frame_results if r["moisture_detection"]["detected"])
        
        if urgent_count > 0:
            return "🚨 URGENT: Critical defects detected in multiple frames - immediate inspection required"
        elif crack_frames > 0 and moisture_frames > 0:
            return "⚠️ Multiple defect types detected - comprehensive assessment recommended"
        elif crack_frames > 0:
            return "📝 Structural issues detected - schedule structural inspection"
        elif moisture_frames > 0:
            return "💧 Moisture issues detected - investigate source and remediation"
        else:
            return "✅ No significant defects detected in video analysis"