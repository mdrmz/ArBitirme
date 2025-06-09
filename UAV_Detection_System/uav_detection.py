import sys
import cv2
import numpy as np
import requests
import json
import os
import time
import datetime
from pathlib import Path
from PySide6.QtCore import (QTimer, Qt, QSize, QThread, Signal, 
                           QObject, QPropertyAnimation, Property,
                           QEasingCurve, QPoint, QRect)
from PySide6.QtGui import (QImage, QPixmap, QFont, QPalette, QColor, 
                          QIcon, QLinearGradient, QBrush, QPainter, 
                          QPen, QRadialGradient, QPainterPath, QFontDatabase)
from PySide6.QtWidgets import (QApplication, QWidget, QLabel, QVBoxLayout,
                             QHBoxLayout, QFrame, QSizePolicy, QGroupBox,
                             QLineEdit, QPushButton, QTextEdit, QMessageBox,
                             QMainWindow, QComboBox, QScrollArea, QSpacerItem,
                             QStackedWidget, QProgressBar, QSlider, QSplitter,
                             QTabWidget, QToolButton, QGridLayout, QFileDialog,
                             QMenu, QSystemTrayIcon, QStyle, QDialog, QCheckBox,
                             QSpinBox, QDoubleSpinBox, QRadioButton, QButtonGroup,
                             QGraphicsDropShadowEffect)
from ultralytics import YOLO
from Sound_Project.Sound import ses, diger

# --- Configuration ---
UAV_MODEL_PATH = "models/yolov11n-face.pt"  # Yeni eğitilen UAV modeli
CAMERA_INDEX = 0
TIMER_INTERVAL_MS = 30
VIDEO_FOLDER = "uav_videos"
JSON_PATH = "uav_info.json"
CONFIDENCE_THRESHOLD = 0.6
SOUND_COOLDOWN = 2.0
LM_STUDIO_BASE_URL = "http://10.52.15.98:40"
LM_STUDIO_URL = f"{LM_STUDIO_BASE_URL}/v1/chat/completions"

# App settings
APP_NAME = "UAV Detection System"
APP_VERSION = "1.0.0"
SETTINGS_FILE = "uav_settings.json"
LOG_FILE = "uav_log.txt"
SCREENSHOT_FOLDER = "uav_screenshots"
RECORDING_FOLDER = "uav_recordings"
MODELS_FOLDER = "uav_models"

# Create necessary folders
for folder in [SCREENSHOT_FOLDER, RECORDING_FOLDER, MODELS_FOLDER, VIDEO_FOLDER]:
    Path(folder).mkdir(exist_ok=True)

# Gradient color definitions for UI
PRIMARY_COLOR = QColor(42, 42, 64)  # Dark blue
SECONDARY_COLOR = QColor(72, 72, 108)  # Medium blue
ACCENT_COLOR = QColor(114, 159, 207)  # Light blue
GRADIENT_START = QColor(35, 35, 55)  # Darker blue
GRADIENT_END = QColor(52, 52, 78)  # Lighter blue
TEXT_COLOR = QColor(240, 240, 250)  # Almost white
HIGHLIGHT_COLOR = QColor(72, 139, 220)  # Bright blue

class VideoThread(QThread):
    """Thread for playing class-specific videos"""
    frame_ready = Signal(np.ndarray)
    error_occurred = Signal(str)
    
    def __init__(self):
        super().__init__()
        self.video_path = None
        self.running = True
        self.cap = None
        
    def set_video(self, path):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.video_path = path
        
    def run(self):
        while self.running:
            if self.video_path and os.path.exists(self.video_path):
                try:
                    self.cap = cv2.VideoCapture(self.video_path)
                    if not self.cap.isOpened():
                        raise Exception(f"Failed to open video: {self.video_path}")
                    
                    while self.running and self.video_path:
                        ret, frame = self.cap.read()
                        if not ret:
                            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            continue
                        
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        self.frame_ready.emit(frame_rgb)
                        time.sleep(0.03)
                    
                    if self.cap is not None:
                        self.cap.release()
                        self.cap = None
                        
                except Exception as e:
                    self.error_occurred.emit(str(e))
                    if self.cap is not None:
                        self.cap.release()
                        self.cap = None
            else:
                time.sleep(0.1)
    
    def stop(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()
        self.wait()

class SoundThread(QThread):
    """Thread for handling TTS announcements"""
    def __init__(self):
        super().__init__()
        self.last_sound_time = 0
        self.last_sound_class = None
        self.running = True
        self.queue = []
        
    def add_to_queue(self, label):
        current_time = time.time()
        if (current_time - self.last_sound_time > SOUND_COOLDOWN or 
            self.last_sound_class != label):
            self.queue.append(label)
            self.last_sound_time = current_time
            self.last_sound_class = label
    
    def run(self):
        while self.running:
            if self.queue:
                label = self.queue.pop(0)
                try:
                    ses(f"UAV tespit edildi: {label}")
                    ses("Hedef kilitlendi")
                except Exception as e:
                    print(f"Sound error: {e}")
            time.sleep(0.1)
    
    def stop(self):
        self.running = False
        self.wait()

class UAVDetectionThread(QThread):
    """Thread for UAV detection using YOLO"""
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
        
        # Pre-initialize camera with optimized settings
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer size
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # Use MJPG codec for faster capture
            self.cap.release()  # Release for later use
        
    def run(self):
        try:
            # Load YOLO model in parallel with camera initialization
            self.model = YOLO(self.model_path)
            
            # Initialize camera with optimized settings
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                raise Exception(f"Failed to open camera {self.camera_index}")
            
            # Set optimized camera settings
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            
            # Warm up camera
            for _ in range(5):
                self.cap.read()
            
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    self.cap.release()
                    time.sleep(0.1)  # Reduced sleep time
                    self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
                    if not self.cap.isOpened():
                        raise Exception("Camera reconnection failed")
                    continue
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Run YOLO inference
                results = self.model(frame_rgb, conf=self.conf_threshold)[0]
                processed_frame = frame_rgb.copy()
                
                # Process detections
                detections = []
                for box in results.boxes:
                    cls = int(box.cls)
                    conf = float(box.conf[0])
                    label = self.model.names[cls]
                    coords = box.xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = coords
                    
                    # Draw bounding box
                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Draw label
                    text = f"{label}: {conf:.2f}"
                    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    
                    cv2.rectangle(processed_frame, (x1, y1 - text_height - baseline - 5), 
                                (x1 + text_width + 5, y1), (0, 200, 0), -1)
                    
                    cv2.putText(processed_frame, text, (x1 + 2, y1 - baseline - 2), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    detections.append({
                        'object_id': len(detections) + 1,
                        'class_name': label,
                        'confidence': conf,
                        'bounding_box': [int(x1), int(y1), int(x2), int(y2)]
                    })
                
                self.frame_ready.emit(processed_frame, detections)
                time.sleep(0.01)
                
        except Exception as e:
            self.error_occurred.emit(str(e))
        finally:
            if self.cap is not None:
                self.cap.release()
    
    def stop(self):
        self.running = False
        self.wait()

class UAVDetectionSystem(QMainWindow):
    """Main application window for UAV detection"""
    def __init__(self):
        super().__init__()
        
        # Setup window
        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}")
        self.resize(1280, 800)
        
        # Initialize data
        self.json_data = self.load_json_data(JSON_PATH)
        self.video_map = self.create_video_map(VIDEO_FOLDER)
        self.current_class = None
        self.confidence_threshold = CONFIDENCE_THRESHOLD
        
        # Initialize recording
        self.recording = False
        self.video_writer = None
        
        # Initialize threads
        self.init_threads()
        
        # Setup UI
        self.init_ui()
        
        # Start detection
        self.detection_thread.start()
        self.video_thread.start()
        self.sound_thread.start()
        
        # Apply style
        self.apply_style()
    
    def init_threads(self):
        """Initialize all worker threads"""
        # Detection thread
        self.detection_thread = UAVDetectionThread(
            UAV_MODEL_PATH, 
            CAMERA_INDEX, 
            self.confidence_threshold
        )
        self.detection_thread.frame_ready.connect(self.update_detection_frame)
        self.detection_thread.error_occurred.connect(self.handle_detection_error)
        
        # Video thread
        self.video_thread = VideoThread()
        self.video_thread.frame_ready.connect(self.update_video_frame)
        self.video_thread.error_occurred.connect(self.handle_video_error)
        
        # Sound thread
        self.sound_thread = SoundThread()
    
    def init_ui(self):
        """Initialize the user interface"""
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Left panel (Camera feed and controls)
        left_panel = QVBoxLayout()
        
        # Camera feed
        camera_frame = QFrame()
        camera_frame.setStyleSheet(f"background-color: {SECONDARY_COLOR.name()}; border-radius: 8px;")
        camera_layout = QVBoxLayout(camera_frame)
        
        self.camera_label = QLabel("Kamera başlatılıyor...")
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setFixedSize(640, 480)
        camera_layout.addWidget(self.camera_label)
        
        # Camera controls
        controls_layout = QHBoxLayout()
        
        self.screenshot_button = QPushButton("Ekran Görüntüsü")
        self.screenshot_button.clicked.connect(self.take_screenshot)
        
        self.record_button = QPushButton("Kayıt Başlat")
        self.record_button.clicked.connect(self.toggle_recording)
        
        controls_layout.addWidget(self.screenshot_button)
        controls_layout.addWidget(self.record_button)
        
        camera_layout.addLayout(controls_layout)
        left_panel.addWidget(camera_frame)
        
        # Right panel (Video and info)
        right_panel = QVBoxLayout()
        
        # Video display
        video_frame = QFrame()
        video_frame.setStyleSheet(f"background-color: {SECONDARY_COLOR.name()}; border-radius: 8px;")
        video_layout = QVBoxLayout(video_frame)
        
        video_title = QLabel("Nesne Görselleştirme")
        video_title.setStyleSheet("""
            font-size: 16px;
            font-weight: bold;
            color: #A0C8E5;
            padding-bottom: 5px;
            border-bottom: 1px solid rgba(114, 159, 207, 120);
        """)
        video_title.setAlignment(Qt.AlignCenter)
        
        self.video_label = QLabel("Nesne tespit edildiğinde görselleştirme burada gösterilecek")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setFixedSize(320, 240)
        
        video_layout.addWidget(video_title)
        video_layout.addWidget(self.video_label)
        right_panel.addWidget(video_frame)
        
        # Info panel
        info_frame = QFrame()
        info_frame.setStyleSheet(f"background-color: {SECONDARY_COLOR.name()}; border-radius: 8px;")
        info_layout = QVBoxLayout(info_frame)
        
        info_title = QLabel("Nesne Bilgisi")
        info_title.setStyleSheet("""
            font-size: 16px;
            font-weight: bold;
            color: #A0C8E5;
            padding-bottom: 5px;
            border-bottom: 1px solid rgba(114, 159, 207, 120);
        """)
        info_title.setAlignment(Qt.AlignCenter)
        
        self.info_label = QLabel("Nesne tespit edildiğinde bilgiler burada gösterilecek")
        self.info_label.setWordWrap(True)
        
        info_layout.addWidget(info_title)
        info_layout.addWidget(self.info_label)
        right_panel.addWidget(info_frame)
        
        # Add panels to main layout
        main_layout.addLayout(left_panel)
        main_layout.addLayout(right_panel)
    
    def update_detection_frame(self, frame, detections):
        """Update the camera feed with detections"""
        if frame is not None:
            # Save frame for recording if enabled
            if self.recording and self.video_writer is not None:
                try:
                    self.video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                except Exception as e:
                    print(f"Recording error: {e}")
            
            # Convert to pixmap and display
            pixmap = QPixmap.fromImage(QImage(frame.data, frame.shape[1], frame.shape[0], 
                                            frame.shape[1] * 3, QImage.Format_RGB888))
            scaled_pixmap = pixmap.scaled(640, 480, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.camera_label.setPixmap(scaled_pixmap)
            self.last_frame = frame
        
        if detections:
            # Get the detection with highest confidence
            best_detection = max(detections, key=lambda x: x['confidence'])
            class_name = best_detection['class_name']
            
            # Update info
            if class_name != self.current_class:
                self.current_class = class_name
                self.update_object_info(class_name)
                
                # Play sound
                try:
                    self.sound_thread.add_to_queue(class_name)
                except Exception as e:
                    print(f"Sound error: {e}")
                
                # Play video if available
                if class_name in self.video_map:
                    try:
                        video_path = self.video_map[class_name]
                        self.video_thread.set_video(video_path)
                        self.video_label.setText(f"{class_name} görselleştirmesi gösteriliyor...")
                    except Exception as e:
                        print(f"Video playback error: {e}")
                        self.video_label.setText(f"{class_name} için video bulunamadı")
                else:
                    self.video_label.setText(f"{class_name} için video bulunamadı")
    
    def update_video_frame(self, frame):
        """Update the video display"""
        if frame is not None:
            pixmap = QPixmap.fromImage(QImage(frame.data, frame.shape[1], frame.shape[0], 
                                            frame.shape[1] * 3, QImage.Format_RGB888))
            scaled_pixmap = pixmap.scaled(320, 240, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.video_label.setPixmap(scaled_pixmap)
    
    def update_object_info(self, class_name):
        """Update object information"""
        if class_name in self.json_data:
            obj_info = self.json_data[class_name]
            info_text = f"""
            <b>{obj_info.get('name', class_name)}</b>
            
            <p><b>Tanım:</b> {obj_info.get('description', 'Bilgi bulunamadı')}</p>
            
            <p><b>Özellikler:</b> {obj_info.get('features', 'Bilgi bulunamadı')}</p>
            
            <p><b>Tehlike Seviyesi:</b> {obj_info.get('threat_level', 'Bilgi bulunamadı')}</p>
            
            <p><b>İlginç Bilgi:</b> {obj_info.get('interesting_facts', 'Bilgi bulunamadı')}</p>
            """
            self.info_label.setText(info_text)
    
    def create_video_map(self, video_folder):
        """Create mapping from class names to video files"""
        video_map = {}
        if not os.path.exists(video_folder):
            return video_map
        for filename in os.listdir(video_folder):
            if filename.endswith(('.mp4', '.avi', '.mov')):
                class_name = os.path.splitext(filename)[0]
                video_map[class_name] = os.path.join(video_folder, filename)
        return video_map
    
    def load_json_data(self, json_path):
        """Load object information from JSON file"""
        try:
            if not os.path.exists(json_path):
                print(f"Warning: JSON file '{json_path}' not found")
                return {}
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if "objects" in data:
                    return data["objects"]
                return data
        except Exception as e:
            print(f"Error loading JSON data: {e}")
            return {}
    
    def take_screenshot(self):
        """Take a screenshot of the current frame"""
        if hasattr(self, 'last_frame'):
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(SCREENSHOT_FOLDER, f"screenshot_{timestamp}.png")
            cv2.imwrite(filename, cv2.cvtColor(self.last_frame, cv2.COLOR_RGB2BGR))
            print(f"Screenshot saved: {filename}")
    
    def toggle_recording(self):
        """Toggle video recording"""
        if not self.recording:
            # Start recording
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(RECORDING_FOLDER, f"recording_{timestamp}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(filename, fourcc, 30.0, (640, 480))
            self.recording = True
            self.record_button.setText("Kayıt Durdur")
            print("Recording started")
        else:
            # Stop recording
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None
            self.recording = False
            self.record_button.setText("Kayıt Başlat")
            print("Recording stopped")
    
    def handle_detection_error(self, error_msg):
        """Handle detection thread errors"""
        print(f"Detection error: {error_msg}")
        self.stop_detection()
    
    def handle_video_error(self, error_msg):
        """Handle video thread errors"""
        print(f"Video error: {error_msg}")
    
    def stop_detection(self):
        """Stop detection thread"""
        if hasattr(self, 'detection_thread'):
            self.detection_thread.stop()
            self.detection_thread.wait()
    
    def apply_style(self):
        """Apply custom style to the application"""
        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: {PRIMARY_COLOR.name()};
            }}
            QLabel {{
                color: {TEXT_COLOR.name()};
                font-size: 14px;
            }}
            QPushButton {{
                background-color: {ACCENT_COLOR.name()};
                color: white;
                border: none;
                border-radius: 5px;
                padding: 8px 12px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {HIGHLIGHT_COLOR.name()};
            }}
        """)
    
    def closeEvent(self, event):
        """Handle window close event"""
        # Stop recording if active
        if self.recording and self.video_writer is not None:
            self.video_writer.release()
        
        # Stop all threads
        self.detection_thread.stop()
        self.video_thread.stop()
        self.sound_thread.stop()
        
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = UAVDetectionSystem()
    window.show()
    sys.exit(app.exec()) 