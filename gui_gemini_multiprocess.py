import sys
import cv2
import numpy as np
from PySide6.QtCore import QTimer, Qt, QSize, QThread, Signal, QProcess
from PySide6.QtGui import QImage, QPixmap, QFont
from PySide6.QtWidgets import (QApplication, QWidget, QLabel, QVBoxLayout,
                             QHBoxLayout, QFrame, QSizePolicy, QGroupBox,
                             QLineEdit, QPushButton, QTextEdit, QMessageBox)
import multiprocessing as mp
import time
import json
import os
import google.generativeai as genai
from ultralytics import YOLO
from Sound_Project.Sound import ses, diger

# --- Configuration ---
MODEL_PATH = "yolov8n.pt"
CAMERA_INDEX = 0
TIMER_INTERVAL_MS = 30
VIDEO_FOLDER = "video"
JSON_PATH = "object_info.json"
CONFIDENCE_THRESHOLD = 0.6
SOUND_COOLDOWN = 2.0

class GeminiAPIThread(QThread):
    response_ready = Signal(str)
    
    def __init__(self, api_key, object_info):
        super().__init__()
        self.api_key = api_key
        self.object_info = object_info
        self.running = True
        
    def run(self):
        try:
            # Configure Gemini API
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel('gemini-pro')
            
            # Create prompt
            prompt = f"""
            Describe this object in detail:
            Class: {self.object_info['class_name']}
            Confidence: {self.object_info['confidence']}
            Location: {self.object_info['bounding_box']}
            """
            
            # Generate response
            response = model.generate_content(prompt)
            self.response_ready.emit(response.text)
            
        except Exception as e:
            self.response_ready.emit(f"Error: {str(e)}")
    
    def stop(self):
        self.running = False
        self.wait()

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
                        'object_id': len(detections) + 1,
                        'class_name': label,
                        'confidence': conf,
                        'bounding_box': [int(x1), int(y1), int(x2), int(y2)]
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
        
        self.setWindowTitle("Object Detection with Gemini API")
        self.setGeometry(100, 100, 1400, 800)
        self.setStyleSheet("background-color: #1C2526;")
        
        # Load data
        self.json_data = self.load_json_data()
        self.video_map = self.create_video_map()
        
        # Initialize threads
        self.detection_thread = DetectionThread(MODEL_PATH, CAMERA_INDEX, CONFIDENCE_THRESHOLD)
        self.video_thread = VideoThread()
        self.sound_thread = SoundThread()
        self.gemini_thread = None
        
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
        self.current_detections = []
    
    def init_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # --- Left Panel ---
        left_panel = QVBoxLayout()
        left_panel.setSpacing(8)
        
        # API Key Input
        api_group = QGroupBox("Gemini API Configuration")
        api_group.setStyleSheet("""
            QGroupBox {
                color: #C9D6DF;
                font-size: 14px;
                font-weight: bold;
                border: 1px solid #52616B;
                border-radius: 6px;
                margin-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 5px 10px;
            }
        """)
        api_layout = QVBoxLayout()
        
        self.api_key_input = QLineEdit()
        self.api_key_input.setPlaceholderText("Enter your Gemini API key")
        self.api_key_input.setStyleSheet("""
            QLineEdit {
                padding: 8px;
                background-color: #2E2E2E;
                color: #E8ECEF;
                border: 1px solid #52616B;
                border-radius: 4px;
            }
        """)
        api_layout.addWidget(self.api_key_input)
        
        self.save_api_button = QPushButton("Save API Key")
        self.save_api_button.setStyleSheet("""
            QPushButton {
                padding: 8px;
                background-color: #52616B;
                color: #E8ECEF;
                border: none;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #657B83;
            }
        """)
        self.save_api_button.clicked.connect(self.save_api_key)
        api_layout.addWidget(self.save_api_button)
        
        api_group.setLayout(api_layout)
        left_panel.addWidget(api_group)
        
        # Main video screen
        self.video_label = QLabel("Waiting for camera...", self)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("""
            background-color: #2E2E2E;
            color: #C9D6DF;
            border: 2px solid #52616B;
            border-radius: 8px;
        """)
        self.video_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        left_panel.addWidget(self.video_label, 1)
        
        # Object list
        objects_group = QGroupBox("Detected Objects")
        objects_group.setStyleSheet("""
            QGroupBox {
                color: #C9D6DF;
                font-size: 14px;
                font-weight: bold;
                border: 1px solid #52616B;
                border-radius: 6px;
                margin-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 5px 10px;
            }
        """)
        objects_layout = QVBoxLayout()
        
        self.objects_text = QTextEdit()
        self.objects_text.setReadOnly(True)
        self.objects_text.setStyleSheet("""
            QTextEdit {
                background-color: #2E2E2E;
                color: #E8ECEF;
                border: 1px solid #52616B;
                border-radius: 4px;
                padding: 8px;
            }
        """)
        objects_layout.addWidget(self.objects_text)
        
        objects_group.setLayout(objects_layout)
        left_panel.addWidget(objects_group)
        
        # --- Right Panel ---
        right_panel = QVBoxLayout()
        right_panel.setSpacing(8)
        
        # Secondary video screen
        self.secondary_video_label = QLabel("Detected object video will be shown here", self)
        self.secondary_video_label.setAlignment(Qt.AlignCenter)
        self.secondary_video_label.setStyleSheet("""
            background-color: #2E2E2E;
            color: #C9D6DF;
            border: 2px solid #52616B;
            border-radius: 8px;
        """)
        self.secondary_video_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        right_panel.addWidget(self.secondary_video_label, 1)
        
        # Gemini API Response
        response_group = QGroupBox("Gemini API Response")
        response_group.setStyleSheet("""
            QGroupBox {
                color: #C9D6DF;
                font-size: 14px;
                font-weight: bold;
                border: 1px solid #52616B;
                border-radius: 6px;
                margin-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 5px 10px;
            }
        """)
        response_layout = QVBoxLayout()
        
        self.response_text = QTextEdit()
        self.response_text.setReadOnly(True)
        self.response_text.setStyleSheet("""
            QTextEdit {
                background-color: #2E2E2E;
                color: #E8ECEF;
                border: 1px solid #52616B;
                border-radius: 4px;
                padding: 8px;
            }
        """)
        response_layout.addWidget(self.response_text)
        
        response_group.setLayout(response_layout)
        right_panel.addWidget(response_group)
        
        # Add panels to main layout
        main_layout.addLayout(left_panel, 1)
        separator = QFrame()
        separator.setFrameShape(QFrame.VLine)
        separator.setStyleSheet("background-color: #52616B;")
        main_layout.addWidget(separator)
        main_layout.addLayout(right_panel, 1)
    
    def save_api_key(self):
        api_key = self.api_key_input.text().strip()
        if api_key:
            try:
                # Test the API key
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel('gemini-pro')
                response = model.generate_content("Test connection")
                
                # Save the API key
                with open("gemini_api_key.txt", "w") as f:
                    f.write(api_key)
                
                QMessageBox.information(self, "Success", "API key saved successfully!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Invalid API key: {str(e)}")
        else:
            QMessageBox.warning(self, "Warning", "Please enter an API key")
    
    def load_api_key(self):
        try:
            with open("gemini_api_key.txt", "r") as f:
                return f.read().strip()
        except:
            return None
    
    def update_main_frame(self, frame, detections):
        if frame is not None:
            self.display_image(frame, self.video_label)
            self.current_detections = detections
            self.update_objects_list()
            self.process_detections(detections)
    
    def update_objects_list(self):
        text = ""
        for detection in self.current_detections:
            text += f"Object {detection['object_id']}: {detection['class_name']} "
            text += f"({detection['confidence']:.2%} confidence) "
            text += f"at {detection['bounding_box']}\n"
        self.objects_text.setText(text)
    
    def update_secondary_frame(self, frame):
        if frame is not None:
            self.display_image(frame, self.secondary_video_label)
    
    def process_detections(self, detections):
        if not detections:
            return
        
        # Get the detection with highest confidence
        best_detection = max(detections, key=lambda x: x['confidence'])
        
        # Play video if available
        if best_detection['class_name'] in self.video_map:
            self.play_video_based_on_class(best_detection['class_name'])
        
        # Play sound
        self.sound_thread.play_sound(best_detection['class_name'])
        
        # Get Gemini API description
        api_key = self.load_api_key()
        if api_key and self.gemini_thread is None:
            self.gemini_thread = GeminiAPIThread(api_key, best_detection)
            self.gemini_thread.response_ready.connect(self.update_gemini_response)
            self.gemini_thread.start()
    
    def update_gemini_response(self, response):
        self.response_text.setText(response)
        self.gemini_thread = None
    
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
        if self.gemini_thread:
            self.gemini_thread.stop()
        event.accept()
    
    def load_json_data(self):
        """Load object information from JSON file or create default if not exists."""
        try:
            if os.path.exists(JSON_PATH):
                with open(JSON_PATH, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                # Create default JSON structure
                default_data = {
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
                # Save default data
                with open(JSON_PATH, 'w', encoding='utf-8') as f:
                    json.dump(default_data, f, indent=4)
                return default_data
        except Exception as e:
            print(f"Error loading JSON data: {e}")
            return {"objects": {}}
    
    def create_video_map(self):
        """Create a mapping of object classes to their video files."""
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

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ObjectDetectionApp()
    window.show()
    sys.exit(app.exec()) 