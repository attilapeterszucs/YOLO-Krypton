"""
YOLO Krypton - Detection Module
Handles all YOLO-related operations including model loading, inference, and result processing
"""

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import torch
from PIL import Image
import json
from datetime import datetime
import config


class YOLODetector:
    """Professional YOLO detector with advanced features"""
    
    def __init__(self, model_path: str = config.DEFAULT_MODEL):
        """Initialize YOLO detector with specified model"""
        self.model_path = Path(config.MODELS_DIR) / model_path
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.class_names = []
        self.colors = {}
        self.load_model(model_path)
        
    def load_model(self, model_name: str):
        """Load YOLO model"""
        try:
            model_path = Path(config.MODELS_DIR) / model_name
            
            # Download model if it doesn't exist
            if not model_path.exists():
                print(f"Downloading {model_name}...")
                self.model = YOLO(model_name)
                # Save the model to models directory
                self.model.save(str(model_path))
            else:
                self.model = YOLO(str(model_path))
            
            # Get class names
            self.class_names = self.model.names
            
            # Generate colors for each class
            self._generate_class_colors()
            
            print(f"Model loaded successfully on {self.device}")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def _generate_class_colors(self):
        """Generate unique colors for each class"""
        np.random.seed(42)
        for i, class_name in enumerate(self.class_names.values()):
            color = tuple(np.random.randint(0, 255, 3).tolist())
            self.colors[class_name] = color
    
    def detect_image(self, image_path: str, confidence: float = 0.5, 
                    iou_threshold: float = 0.45) -> Dict[str, Any]:
        """Perform detection on a single image"""
        try:
            # Run inference
            results = self.model(
                image_path,
                conf=confidence,
                iou=iou_threshold,
                device=self.device
            )
            
            # Process results
            detections = self._process_results(results[0])
            
            # Draw bounding boxes
            annotated_image = self._draw_detections(
                cv2.imread(image_path),
                detections
            )
            
            return {
                'success': True,
                'detections': detections,
                'annotated_image': annotated_image,
                'total_objects': len(detections),
                'processing_time': results[0].speed['inference']
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'detections': [],
                'annotated_image': None
            }
    
    def detect_video(self, video_path: str, confidence: float = 0.5,
                    iou_threshold: float = 0.45, callback=None) -> Dict[str, Any]:
        """Perform detection on video with frame callback"""
        try:
            cap = cv2.VideoCapture(video_path)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            all_detections = []
            frame_count = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Run detection on frame
                results = self.model(
                    frame,
                    conf=confidence,
                    iou=iou_threshold,
                    device=self.device
                )
                
                # Process results
                detections = self._process_results(results[0])
                annotated_frame = self._draw_detections(frame, detections)
                
                # Store frame detections
                all_detections.append({
                    'frame': frame_count,
                    'detections': detections
                })
                
                # Callback for UI update
                if callback:
                    callback(annotated_frame, frame_count, total_frames)
                
                frame_count += 1
            
            cap.release()
            
            return {
                'success': True,
                'total_frames': frame_count,
                'fps': fps,
                'all_detections': all_detections
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def detect_webcam(self, confidence: float = 0.5, iou_threshold: float = 0.45,
                     callback=None, stop_event=None):
        """Real-time detection from webcam"""
        try:
            cap = cv2.VideoCapture(0)
            
            while cap.isOpened():
                if stop_event and stop_event.is_set():
                    break
                    
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Run detection
                results = self.model(
                    frame,
                    conf=confidence,
                    iou=iou_threshold,
                    device=self.device
                )
                
                # Process results
                detections = self._process_results(results[0])
                annotated_frame = self._draw_detections(frame, detections)
                
                # Callback for UI update
                if callback:
                    callback(annotated_frame, detections)
            
            cap.release()
            
        except Exception as e:
            print(f"Webcam error: {e}")
    
    def _process_results(self, result) -> List[Dict]:
        """Process YOLO results into structured format"""
        detections = []
        
        if result.boxes is not None:
            boxes = result.boxes
            for i in range(len(boxes)):
                box = boxes[i]
                detection = {
                    'class_id': int(box.cls),
                    'class_name': self.class_names[int(box.cls)],
                    'confidence': float(box.conf),
                    'bbox': box.xyxy[0].tolist(),  # [x1, y1, x2, y2]
                    'center': [(box.xyxy[0][0] + box.xyxy[0][2]) / 2,
                              (box.xyxy[0][1] + box.xyxy[0][3]) / 2]
                }
                detections.append(detection)
        
        return detections
    
    def _draw_detections(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw bounding boxes and labels on image"""
        annotated = image.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = map(int, detection['bbox'])
            class_name = detection['class_name']
            confidence = detection['confidence']
            color = self.colors.get(class_name, (0, 255, 0))
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw label background
            label = f"{class_name} {confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            label_y = y1 - 10 if y1 - 10 > 10 else y1 + 20
            
            cv2.rectangle(
                annotated,
                (x1, label_y - label_size[1] - 5),
                (x1 + label_size[0], label_y + 5),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                annotated,
                label,
                (x1, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
        
        return annotated
    
    def export_results(self, detections: List[Dict], format: str = "JSON",
                      output_path: Optional[str] = None) -> str:
        """Export detection results in various formats"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if not output_path:
            output_path = config.OUTPUT_DIR / f"detections_{timestamp}"
        
        if format == "JSON":
            output_file = f"{output_path}.json"
            with open(output_file, 'w') as f:
                json.dump(detections, f, indent=2)
                
        elif format == "CSV":
            import csv
            output_file = f"{output_path}.csv"
            with open(output_file, 'w', newline='') as f:
                if detections:
                    writer = csv.DictWriter(f, fieldnames=detections[0].keys())
                    writer.writeheader()
                    writer.writerows(detections)
                    
        elif format == "TXT":
            output_file = f"{output_path}.txt"
            with open(output_file, 'w') as f:
                for det in detections:
                    f.write(f"{det['class_name']}: {det['confidence']:.2f} "
                           f"at [{det['bbox'][0]:.1f}, {det['bbox'][1]:.1f}, "
                           f"{det['bbox'][2]:.1f}, {det['bbox'][3]:.1f}]\n")
                           
        elif format == "YOLO":
            output_file = f"{output_path}_yolo.txt"
            with open(output_file, 'w') as f:
                for det in detections:
                    # Convert to YOLO format (normalized)
                    f.write(f"{det['class_id']} {det['center'][0]} {det['center'][1]} "
                           f"{det['bbox'][2] - det['bbox'][0]} "
                           f"{det['bbox'][3] - det['bbox'][1]}\n")
        
        return output_file
    
    def get_statistics(self, detections: List[Dict]) -> Dict[str, Any]:
        """Generate statistics from detections"""
        if not detections:
            return {'total_objects': 0, 'class_distribution': {}}
        
        class_counts = {}
        confidence_scores = []
        
        for det in detections:
            class_name = det['class_name']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            confidence_scores.append(det['confidence'])
        
        return {
            'total_objects': len(detections),
            'unique_classes': len(class_counts),
            'class_distribution': class_counts,
            'average_confidence': np.mean(confidence_scores),
            'min_confidence': np.min(confidence_scores),
            'max_confidence': np.max(confidence_scores)
        }
