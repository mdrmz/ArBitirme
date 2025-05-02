import sys
import cv2
import numpy as np
from PySide6.QtCore import QTimer, Qt, QSize, QThread, Signal
from PySide6.QtGui import QImage, QPixmap, QFont
from PySide6.QtWidgets import (QApplication, QWidget, QLabel, QVBoxLayout,
                             QHBoxLayout, QFrame, QSizePolicy, QGroupBox)
import multiprocessing as mp
import time
import json
import os
from ultralytics import YOLO
from Sound_Project.Sound import ses, diger

# --- Configuration ---
MODEL_PATH = "yolov8n.pt"
CAMERA_INDEX = 0
TIMER_INTERVAL_MS = 30
VIDEO_FOLDER = "video"
JSON_PATH = "object_info.json"
CONFIDENCE_THRESHOLD = 0.6
SOUND_COOLDOWN = 2.0  # Seconds before repeating the same sound

# Create default object_info.json if it doesn't exist
DEFAULT_OBJECT_INFO = {
    "objects": {
        "person": {
            "name": "Person",
            "description": "A human being",
            "sound": "person.mp3",
            "video": "person.mp4"
        },
        "car": {
            "name": "Car",
            "description": "A motor vehicle",
            "sound": "car.mp3",
            "video": "car.mp4"
        },
        "dog": {
            "name": "Dog",
            "description": "A domestic animal",
            "sound": "dog.mp3",
            "video": "dog.mp4"
        }
    }
}

if not os.path.exists(JSON_PATH):
    with open(JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(DEFAULT_OBJECT_INFO, f, indent=4)

class DetectionThread(QThread):
    frame_ready = Signal(np.ndarray, list)
    
    def __init__(self, model_path, camera_index, conf_threshold):
        super().__init__()
        self.model_path = model_path
        self.camera_index = camera_index
        self.conf_threshold = conf_threshold
        self.running = True
        
    def run(self):
        try:
            model = YOLO(self.model_path)
            cap = cv2.VideoCapture(self.camera_index)
            
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    continue
                
                # Run detection
                results = model(frame, conf=self.conf_threshold)[0]
                detections = []
                
                for box in results.boxes:
                    cls = int(box.cls)
                    conf = float(box.conf[0])
                    label = model.names[cls]
                    coords = box.xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = coords
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    text = f"{label}: {conf:.2f}"
                    cv2.putText(frame, text, (x1, y1 - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    detections.append({
                        'label': label,
                        'conf': conf,
                        'coords': (x1, y1, x2, y2)
                    })
                
                self.frame_ready.emit(frame, detections)
                
        except Exception as e:
            print(f"Error in detection thread: {e}")
        finally:
            if 'cap' in locals():
                cap.release()
    
    def stop(self):
        self.running = False
        self.wait()

class VideoThread(QThread):
    frame_ready = Signal(np.ndarray)
    
    def __init__(self):
        super().__init__()
        self.video_path = None
        self.running = True
        
    def set_video(self, path):
        self.video_path = path
        
    def run(self):
        while self.running:
            if self.video_path:
                try:
                    cap = cv2.VideoCapture(self.video_path)
                    while self.running and self.video_path:
                        ret, frame = cap.read()
                        if not ret:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            continue
                        self.frame_ready.emit(frame)
                    cap.release()
                except Exception as e:
                    print(f"Error in video thread: {e}")
            else:
                time.sleep(0.01)
    
    def stop(self):
        self.running = False
        self.wait()

class SoundThread(QThread):
    def __init__(self):
        super().__init__()
        self.last_sound_time = 0
        self.last_sound_class = None
        self.running = True
        
    def play_sound(self, label):
        current_time = time.time()
        if (current_time - self.last_sound_time > SOUND_COOLDOWN or 
            self.last_sound_class != label):
            ses(f"Detected {label}")
            self.last_sound_time = current_time
            self.last_sound_class = label
    
    def stop(self):
        self.running = False
        self.wait()

class ObjectDetectionApp(QWidget):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Object Detection GUI")
        self.setGeometry(100, 100, 1200, 700)
        self.setStyleSheet("background-color: #1C2526;")
        
        # Load data
        self.json_data = self.load_json_data()
        self.video_map = self.create_video_map()
        
        # Initialize threads
        self.detection_thread = DetectionThread(MODEL_PATH, CAMERA_INDEX, CONFIDENCE_THRESHOLD)
        self.video_thread = VideoThread()
        self.sound_thread = SoundThread()
        
        # Connect signals
        self.detection_thread.frame_ready.connect(self.update_main_frame)
        self.video_thread.frame_ready.connect(self.update_secondary_frame)
        
        # Start threads
        self.detection_thread.start()
        self.video_thread.start()
        self.sound_thread.start()
        
        # Initialize UI
        self.init_ui()
        
        self.current_playing_class = None
    
    def load_json_data(self):
        try:
            with open(JSON_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get("objects", {})
        except Exception as e:
            print(f"Error loading JSON data: {e}")
            return {}
    
    def create_video_map(self):
        video_map = {}
        try:
            if os.path.exists(VIDEO_FOLDER):
                for filename in os.listdir(VIDEO_FOLDER):
                    if filename.endswith(('.mp4', '.avi', '.mov')):
                        class_name = os.path.splitext(filename)[0]
                        video_map[class_name] = os.path.join(VIDEO_FOLDER, filename)
        except Exception as e:
            print(f"Error creating video map: {e}")
        return video_map
    
    def init_ui(self):
        # ... (same UI initialization code as before) ...
        pass
    
    def update_main_frame(self, frame, detections):
        if frame is not None:
            self.display_image(frame, self.video_label)
            self.process_detections(detections)
    
    def update_secondary_frame(self, frame):
        if frame is not None:
            self.display_image(frame, self.secondary_video_label)
    
    def process_detections(self, detections):
        if not detections:
            return
        
        # Get the detection with highest confidence
        best_detection = max(detections, key=lambda x: x['conf'])
        label = best_detection['label']
        conf = best_detection['conf']
        
        # Update features label
        features_text = f"Detected: {label}\nConfidence: {conf:.2f}"
        self.features_label.setText(features_text)
        
        # Update JSON information
        if label in self.json_data:
            info = self.json_data[label]
            self.json_info_label.setText(
                f"Name: {info.get('name', label)}\n"
                f"Description: {info.get('description', 'No description available')}"
            )
        else:
            self.json_info_label.setText(f"No additional information available for {label}")
        
        # Play video if available
        if label in self.video_map:
            self.play_video_based_on_class(label)
        
        # Play sound
        self.sound_thread.play_sound(label)
    
    def play_video_based_on_class(self, object_class):
        if object_class != self.current_playing_class:
            if object_class in self.video_map:
                self.video_thread.set_video(self.video_map[object_class])
                self.current_playing_class = object_class
            else:
                self.stop_secondary_video()
    
    def stop_secondary_video(self):
        self.video_thread.set_video(None)
        self.current_playing_class = None
        self.clear_label(self.secondary_video_label)
    
    def display_image(self, frame, label_widget):
        if frame is not None:
            # Convert frame to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert to QImage
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            # Scale to fit label while maintaining aspect ratio
            scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
                label_widget.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            
            label_widget.setPixmap(scaled_pixmap)
    
    def clear_label(self, label_widget, text=""):
        label_widget.clear()
        if text:
            label_widget.setText(text)
    
    def closeEvent(self, event):
        # Stop all threads
        self.detection_thread.stop()
        self.video_thread.stop()
        self.sound_thread.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ObjectDetectionApp()
    window.show()
    sys.exit(app.exec()) 