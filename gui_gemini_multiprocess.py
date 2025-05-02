import sys
import cv2
import numpy as np
from PySide6.QtCore import QTimer, Qt, QSize, QThread, Signal, QObject
from PySide6.QtGui import QImage, QPixmap, QFont, QPalette, QColor, QIcon
from PySide6.QtWidgets import (QApplication, QWidget, QLabel, QVBoxLayout,
                             QHBoxLayout, QFrame, QSizePolicy, QGroupBox,
                             QLineEdit, QPushButton, QTextEdit, qQMessageBox,
                             QMainWindow)
import time
import json
import os
import google.generativeai as genai
from ultralytics import YOLO
from Sound_Project.Sound import ses
import qdarkstyle

# --- Configuration ---
MODEL_PATH = "yolov8n.pt"
CAMERA_INDEX = 0
TIMER_INTERVAL_MS = 30
VIDEO_FOLDER = "video"
JSON_PATH = "object_info.json"
CONFIDENCE_THRESHOLD = 0.6
SOUND_COOLDOWN = 2.0

class DetectionThread(QThread):
    frame_ready = Signal(np.ndarray, list)
    error_occurred = Signal(str)
    
    def __init__(self, model_path, camera_index, conf_threshold):
        super().__init__()
        self.model_path = model_path
        self.camera_index = camera_index
        self.conf_threshold = conf_threshold
        self.running = True
        self.model = None
        self.cap = None
        
    def run(self):
        try:
            # Initialize YOLO model
            self.model = YOLO(self.model_path)
            
            # Initialize camera
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                raise Exception("Failed to open camera")
            
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    continue
                
                # Run detection
                results = self.model(frame, conf=self.conf_threshold)[0]
                detections = []
                
                # Process detections
                for box in results.boxes:
                    cls = int(box.cls)
                    conf = float(box.conf[0])
                    label = self.model.names[cls]
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
                time.sleep(0.01)  # Prevent CPU overload
                
        except Exception as e:
            self.error_occurred.emit(str(e))
        finally:
            if self.cap is not None:
                self.cap.release()
    
    def stop(self):
        self.running = False
        self.wait()

class VideoThread(QThread):
    frame_ready = Signal(np.ndarray)
    error_occurred = Signal(str)
    
    def __init__(self):
        super().__init__()
        self.video_path = None
        self.running = True
        self.cap = None
        
    def set_video(self, path):
        self.video_path = path
        
    def run(self):
        while self.running:
            if self.video_path:
                try:
                    self.cap = cv2.VideoCapture(self.video_path)
                    if not self.cap.isOpened():
                        raise Exception(f"Failed to open video: {self.video_path}")
                    
                    while self.running and self.video_path:
                        ret, frame = self.cap.read()
                        if not ret:
                            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            continue
                        self.frame_ready.emit(frame)
                        time.sleep(0.01)  # Control video playback speed
                    
                    if self.cap is not None:
                        self.cap.release()
                        
                except Exception as e:
                    self.error_occurred.emit(str(e))
            else:
                time.sleep(0.01)
    
    def stop(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()
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
            try:
                ses(f"Detected {label}")
                self.last_sound_time = current_time
                self.last_sound_class = label
            except Exception as e:
                print(f"Sound error: {e}")
    
    def stop(self):
        self.running = False
        self.wait()

class GeminiThread(QThread):
    response_ready = Signal(str)
    error_occurred = Signal(str)
    
    def __init__(self, api_key, object_info):
        super().__init__()
        self.api_key = api_key
        self.object_info = object_info
        
    def run(self):
        try:
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel('gemini-pro')
            
            prompt = f"""
            Describe this object in detail:
            Class: {self.object_info['class_name']}
            Confidence: {self.object_info['confidence']}
            Location: {self.object_info['bounding_box']}
            """
            
            response = model.generate_content(prompt)
            self.response_ready.emit(response.text)
        except Exception as e:
            self.error_occurred.emit(str(e))

class ObjectDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Vision Assistant")
        self.setGeometry(100, 100, 1600, 900)
        
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
        self.detection_thread.error_occurred.connect(self.handle_detection_error)
        self.video_thread.frame_ready.connect(self.update_secondary_frame)
        self.video_thread.error_occurred.connect(self.handle_video_error)
        
        # Start threads
        self.detection_thread.start()
        self.video_thread.start()
        self.sound_thread.start()
        
        # Initialize UI
        self.init_ui()
        
        self.current_playing_class = None
        self.current_detections = []
        
        # Apply modern style
        self.apply_modern_style()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Left Panel
        left_panel = QVBoxLayout()
        
        # API Configuration
        api_group = QGroupBox("AI Configuration")
        api_layout = QVBoxLayout()
        
        self.api_key_input = QLineEdit()
        self.api_key_input.setPlaceholderText("Enter your Gemini API key")
        api_layout.addWidget(self.api_key_input)
        
        self.save_api_button = QPushButton("Save API Key")
        self.save_api_button.clicked.connect(self.save_api_key)
        api_layout.addWidget(self.save_api_button)
        
        api_group.setLayout(api_layout)
        left_panel.addWidget(api_group)
        
        # Main Video Display
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("""
            QLabel {
                background-color: #2E2E2E;
                border: 2px solid #52616B;
                border-radius: 10px;
                padding: 10px;
            }
        """)
        left_panel.addWidget(self.video_label, 1)
        
        # Detection Results
        results_group = QGroupBox("Detection Results")
        results_layout = QVBoxLayout()
        
        self.objects_text = QTextEdit()
        self.objects_text.setReadOnly(True)
        results_layout.addWidget(self.objects_text)
        
        results_group.setLayout(results_layout)
        left_panel.addWidget(results_group)
        
        # Right Panel
        right_panel = QVBoxLayout()
        
        # Secondary Video Display
        self.secondary_video_label = QLabel()
        self.secondary_video_label.setAlignment(Qt.AlignCenter)
        self.secondary_video_label.setMinimumSize(640, 480)
        self.secondary_video_label.setStyleSheet("""
            QLabel {
                background-color: #2E2E2E;
                border: 2px solid #52616B;
                border-radius: 10px;
                padding: 10px;
            }
        """)
        right_panel.addWidget(self.secondary_video_label, 1)
        
        # Gemini Response
        response_group = QGroupBox("AI Description")
        response_layout = QVBoxLayout()
        
        self.response_text = QTextEdit()
        self.response_text.setReadOnly(True)
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

    def apply_modern_style(self):
        self.setStyleSheet(qdarkstyle.load_stylesheet())
        
        # Set window properties
        self.setWindowFlags(Qt.Window | Qt.WindowMinMaxButtonsHint)
        
        # Apply styles to widgets
        style = """
            QGroupBox {
                color: #C9D6DF;
                font-size: 14px;
                font-weight: bold;
                border: 1px solid #52616B;
                border-radius: 8px;
                margin-top: 15px;
                padding: 15px;
            }
            QLineEdit {
                padding: 10px;
                background-color: #2E2E2E;
                color: #E8ECEF;
                border: 1px solid #52616B;
                border-radius: 6px;
                font-size: 13px;
            }
            QPushButton {
                padding: 10px;
                background-color: #52616B;
                color: #E8ECEF;
                border: none;
                border-radius: 6px;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #657B83;
            }
            QTextEdit {
                background-color: #2E2E2E;
                color: #E8ECEF;
                border: 1px solid #52616B;
                border-radius: 6px;
                padding: 10px;
                font-size: 13px;
            }
        """
        self.setStyleSheet(self.styleSheet() + style)

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

    def display_image(self, frame, label_widget):
        if frame is not None:
            try:
                # Get the label size
                label_size = label_widget.size()
                
                # Calculate the aspect ratio
                h, w = frame.shape[:2]
                aspect_ratio = w / h
                
                # Calculate new dimensions while maintaining aspect ratio
                if label_size.width() / label_size.height() > aspect_ratio:
                    new_height = label_size.height()
                    new_width = int(new_height * aspect_ratio)
                else:
                    new_width = label_size.width()
                    new_height = int(new_width / aspect_ratio)
                
                # Resize frame using cv2 for better performance
                resized_frame = cv2.resize(frame, (new_width, new_height), 
                                        interpolation=cv2.INTER_AREA)
                
                # Convert frame to RGB
                rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                
                # Convert to QImage
                bytes_per_line = 3 * new_width
                qt_image = QImage(rgb_frame.data, new_width, new_height, 
                                bytes_per_line, QImage.Format_RGB888)
                
                # Create pixmap and set it to label
                pixmap = QPixmap.fromImage(qt_image)
                label_widget.setPixmap(pixmap)
                
            except Exception as e:
                print(f"Error displaying image: {e}")

    def process_detections(self, detections):
        if not detections:
            self.stop_secondary_video()
            return
        
        best_detection = max(detections, key=lambda x: x['confidence'])
        
        if best_detection['class_name'] in self.video_map:
            self.play_video_based_on_class(best_detection['class_name'])
        
        self.sound_thread.play_sound(best_detection['class_name'])
        
        api_key = self.load_api_key()
        if api_key and self.gemini_thread is None:
            self.gemini_thread = GeminiThread(api_key, best_detection)
            self.gemini_thread.response_ready.connect(self.update_gemini_response)
            self.gemini_thread.error_occurred.connect(self.handle_gemini_error)
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
        self.secondary_video_label.clear()

    def handle_detection_error(self, error_msg):
        QMessageBox.critical(self, "Detection Error", f"Error in detection: {error_msg}")

    def handle_video_error(self, error_msg):
        QMessageBox.critical(self, "Video Error", f"Error in video playback: {error_msg}")

    def handle_gemini_error(self, error_msg):
        QMessageBox.critical(self, "Gemini API Error", f"Error in Gemini API: {error_msg}")

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

    def load_json_data(self):
        try:
            if os.path.exists(JSON_PATH):
                with open(JSON_PATH, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
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
                with open(JSON_PATH, 'w', encoding='utf-8') as f:
                    json.dump(default_data, f, indent=4)
                return default_data
        except Exception as e:
            print(f"Error loading JSON data: {e}")
            return {"objects": {}}

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

    def closeEvent(self, event):
        # Stop all threads
        self.detection_thread.stop()
        self.video_thread.stop()
        self.sound_thread.stop()
        if self.gemini_thread:
            self.gemini_thread.wait()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ObjectDetectionApp()
    window.show()
    sys.exit(app.exec()) 