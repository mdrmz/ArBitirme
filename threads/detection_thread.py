import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal
from ultralytics import YOLO
import time

class DetectionThread(QThread):
    """Thread for running YOLO detection"""
    frame_ready = Signal(object, list)  # frame, detections
    error_occurred = Signal(str)
    
    def __init__(self, model_path, camera_index=0, conf_threshold=0.6):
        super().__init__()
        self.model_path = model_path
        self.camera_index = camera_index
        self.conf_threshold = conf_threshold
        self.running = True
        self.cap = None
        self.model = None
        self.current_frame = None
    
    def run(self):
        """Main thread loop"""
        try:
            # Initialize camera
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                raise Exception("Kamera başlatılamadı")
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            # Load YOLO model with retry
            max_retries = 3
            retry_count = 0
            while retry_count < max_retries:
                try:
                    self.model = YOLO(self.model_path)
                    break
                except Exception as e:
                    retry_count += 1
                    if retry_count == max_retries:
                        raise Exception(f"Model yüklenemedi: {str(e)}")
                    time.sleep(1)  # Wait before retry
            
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    raise Exception("Kamera görüntüsü alınamadı")
                
                # Convert frame to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.current_frame = frame_rgb
                
                try:
                    # Run YOLO inference
                    results = self.model(frame_rgb, conf=self.conf_threshold)[0]
                    
                    # Process detections
                    detections = []
                    for box in results.boxes:
                        class_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        class_name = results.names[class_id]
                        
                        detections.append({
                            'class_name': class_name,
                            'confidence': conf,
                            'box': box.xyxy[0].tolist()
                        })
                    
                    # Draw boxes and labels
                    for det in detections:
                        x1, y1, x2, y2 = map(int, det['box'])
                        cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"{det['class_name']} {det['confidence']:.2f}"
                        cv2.putText(frame_rgb, label, (x1, y1 - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Emit frame and detections
                    self.frame_ready.emit(frame_rgb, detections)
                    
                except Exception as e:
                    self.error_occurred.emit(f"Model çalıştırma hatası: {str(e)}")
                    time.sleep(0.1)  # Prevent tight loop on error
                
        except Exception as e:
            self.error_occurred.emit(str(e))
        finally:
            if self.cap is not None:
                self.cap.release()
    
    def stop(self):
        """Stop the thread"""
        self.running = False
        self.wait() 