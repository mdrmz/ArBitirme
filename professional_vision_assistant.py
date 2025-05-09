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
MODEL_PATH = "yolov8n.pt"
FACE_MODEL_PATH = "yolov11n-face.pt"
CAMERA_INDEX = 0
TIMER_INTERVAL_MS = 30
VIDEO_FOLDER = "video"
JSON_PATH = "object_info.json"
CONFIDENCE_THRESHOLD = 0.6
SOUND_COOLDOWN = 2.0
LM_STUDIO_BASE_URL = "http://10.52.15.98:40"
LM_STUDIO_URL = f"{LM_STUDIO_BASE_URL}/v1/chat/completions"

# App settings
APP_NAME = "Professional Vision Assistant"
APP_VERSION = "1.0.0"
SETTINGS_FILE = "app_settings.json"
LOG_FILE = "app_log.txt"
SCREENSHOT_FOLDER = "screenshots"
RECORDING_FOLDER = "recordings"
MODELS_FOLDER = "models"

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

# Custom fonts
FONT_FAMILY = "Segoe UI"  # Windows default font
FONT_SIZES = {
    "small": 10,
    "medium": 12,
    "large": 14,
    "xlarge": 16,
    "xxlarge": 20,
    "title": 24
}

# Load custom fonts if available
def load_custom_fonts():
    font_dir = Path("fonts")
    if font_dir.exists():
        for font_file in font_dir.glob("*.ttf"):
            QFontDatabase.addApplicationFont(str(font_file))

load_custom_fonts()

# Helper functions for UI/Models
def get_available_models():
    """Get available LM Studio models"""
    try:
        response = requests.get(f"{LM_STUDIO_BASE_URL}/v1/models", timeout=5)
        if response.status_code == 200:
            models = response.json()
            return [model['id'] for model in models.get('data', [])]
        return []
    except Exception as e:
        # Bağlantı hatalarını sessizce logla
        log_message(f"LM Studio model list error: {e}", "ERROR")
        return []

def load_json_data(json_path):
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

def create_video_map(video_folder):
    """Create mapping from class names to video files"""
    video_map = {}
    if not os.path.exists(video_folder):
        return video_map
    for filename in os.listdir(video_folder):
        if filename.endswith(('.mp4', '.avi', '.mov')):
            class_name = os.path.splitext(filename)[0]
            video_map[class_name] = os.path.join(video_folder, filename)
    return video_map

def convert_cv_qt(cv_img):
    """Convert OpenCV image to QPixmap"""
    h, w, ch = cv_img.shape
    bytes_per_line = ch * w
    convert_to_qt_format = QImage(cv_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(convert_to_qt_format)

def save_screenshot(frame, folder=SCREENSHOT_FOLDER):
    """Save a screenshot of the current frame"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(folder, f"screenshot_{timestamp}.png")
    cv2.imwrite(filename, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    return filename

def log_message(message, level="INFO"):
    """Log a message to the log file"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] [{level}] {message}\n")

def load_settings():
    """Load application settings from JSON file"""
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            log_message(f"Error loading settings: {e}", "ERROR")
    return {
        "camera_index": CAMERA_INDEX,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "sound_enabled": True,
        "video_enabled": True,
        "dark_mode": True,
        "font_size": "medium",
        "language": "tr",
        "auto_save_screenshots": False,
        "recording_enabled": False
    }

def save_settings(settings):
    """Save application settings to JSON file"""
    try:
        with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
            json.dump(settings, f, indent=4)
    except Exception as e:
        log_message(f"Error saving settings: {e}", "ERROR")

def get_available_cameras():
    """Get list of available camera devices"""
    available_cameras = []
    for i in range(10):  # Check first 10 indices
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available_cameras.append(i)
            cap.release()
    return available_cameras

def create_recording_writer(folder=RECORDING_FOLDER):
    """Create a video writer for recording"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(folder, f"recording_{timestamp}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(filename, fourcc, 30.0, (640, 480))

def apply_style_sheet(widget, style_type="default"):
    """Apply a predefined style sheet to a widget"""
    styles = {
        "default": f"""
            QWidget {{
                font-family: {FONT_FAMILY};
                font-size: {FONT_SIZES['medium']}px;
                color: {TEXT_COLOR.name()};
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
            QLineEdit {{
                background-color: {SECONDARY_COLOR.name()};
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px;
            }}
            QComboBox {{
                background-color: {SECONDARY_COLOR.name()};
                color: white;
                border: none;
                border-radius: 4px;
                padding: 5px;
            }}
            QSlider::groove:horizontal {{
                border: 1px solid #999999;
                height: 8px;
                background: {SECONDARY_COLOR.name()};
                margin: 2px 0;
                border-radius: 4px;
            }}
            QSlider::handle:horizontal {{
                background: {ACCENT_COLOR.name()};
                border: 1px solid #5c5c5c;
                width: 18px;
                margin: -2px 0;
                border-radius: 9px;
            }}
        """,
        "dark": f"""
            QWidget {{
                background-color: {PRIMARY_COLOR.name()};
                color: {TEXT_COLOR.name()};
            }}
        """,
        "light": """
            QWidget {
                background-color: #F5F5F5;
                color: #333333;
            }
        """
    }
    widget.setStyleSheet(styles.get(style_type, styles["default"]))

def create_tooltip(text, parent=None):
    """Create a custom tooltip widget"""
    tooltip = QLabel(text, parent)
    tooltip.setStyleSheet("""
        background-color: rgba(0, 0, 0, 180);
        color: white;
        border-radius: 4px;
        padding: 5px;
        font-size: 12px;
    """)
    tooltip.setAlignment(Qt.AlignCenter)
    return tooltip

def show_notification(title, message, icon=QSystemTrayIcon.Information):
    """Show a system tray notification"""
    if QSystemTrayIcon.isSystemTrayAvailable():
        tray = QSystemTrayIcon()
        tray.setIcon(QApplication.style().standardIcon(QStyle.SP_ComputerIcon))
        tray.show()
        tray.showMessage(title, message, icon, 3000)

class DetectionThread(QThread):
    """Thread for object detection using YOLO"""
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
            # Load YOLO model
            self.model = YOLO(self.model_path)
            
            # Kamera bağlantısını dene
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)  # DirectShow kullan
            if not self.cap.isOpened():
                raise Exception(f"Failed to open camera {self.camera_index}")
            
            # Kamera ayarlarını optimize et
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    # Kamera hatası durumunda yeniden bağlanmayı dene
                    self.cap.release()
                    time.sleep(1)  # 1 saniye bekle
                    self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
                    if not self.cap.isOpened():
                        raise Exception("Camera reconnection failed")
                    continue
                
                # Convert BGR to RGB (for Qt display)
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
                    
                    # Draw enhanced bounding box with gradient fill
                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Draw label with enhanced background
                    text = f"{label}: {conf:.2f}"
                    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    
                    # Background for text
                    cv2.rectangle(processed_frame, (x1, y1 - text_height - baseline - 5), 
                                (x1 + text_width + 5, y1), (0, 200, 0), -1)
                    
                    # Text
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
                        
                        # Convert BGR to RGB for Qt
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        self.frame_ready.emit(frame_rgb)
                        time.sleep(0.03)  # ~30 FPS
                    
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
                    ses(f"object detected: {label} ")
                    ses("locked on targets")
                except Exception as e:
                    print(f"Sound error: {e}")
            time.sleep(0.1)
    
    def stop(self):
        self.running = False
        self.wait()

class LMStudioChatThread(QThread):
    """Thread for handling LM Studio chat interactions"""
    response_ready = Signal(str, str)  # response, query_type
    error_occurred = Signal(str)
    
    def __init__(self):
        super().__init__()
        self.running = True
        self.queue = []
        self.model_name = None
        self.json_data = {}
        self.connection_error = False
    
    def set_model(self, model_name):
        self.model_name = model_name
        
    def set_json_data(self, json_data):
        self.json_data = json_data
    
    def add_query(self, query_type, class_name, custom_query=None):
        """Add a query to process"""
        self.queue.append({
            'type': query_type,  # 'object_info', 'custom'
            'class_name': class_name,
            'custom_query': custom_query
        })
    
    def run(self):
        while self.running:
            if self.queue and self.model_name:
                query_data = self.queue.pop(0)
                query_type = query_data['type']
                class_name = query_data['class_name']
                
                try:
                    # Önce JSON'dan hızlı yanıt ver
                    if query_type == 'object_info':
                        object_info = self.get_object_info(class_name)
                        if object_info:
                            # JSON'dan hızlı yanıt
                            quick_response = self.get_quick_response(class_name, object_info)
                            self.response_ready.emit(quick_response, query_type)
                    else:
                        custom_query = query_data['custom_query']
                        # JSON'dan hızlı yanıt
                        quick_response = self.get_custom_quick_response(class_name, custom_query)
                        self.response_ready.emit(quick_response, query_type)
                    
                    # Arka planda LM Studio'dan detaylı yanıt al
                    self.get_detailed_response(query_type, class_name, query_data.get('custom_query'))
                    
                except Exception as e:
                    self.error_occurred.emit(str(e))
            
            time.sleep(0.1)
    
    def get_quick_response(self, class_name, object_info):
        """JSON verilerinden hızlı yanıt oluştur"""
        if class_name in self.json_data:
            obj_info = self.json_data[class_name]
            return f"""
            <b>{obj_info.get('name', class_name)}</b>
            
            <p><b>Tanım:</b> {obj_info.get('description', 'Bilgi bulunamadı')}</p>
            
            <p><b>Özellikler:</b> {obj_info.get('features', 'Bilgi bulunamadı')}</p>
            
            <p><b>Kullanım:</b> {obj_info.get('usage', 'Bilgi bulunamadı')}</p>
            
            <p><b>İlginç Bilgi:</b> {obj_info.get('interesting_facts', 'Bilgi bulunamadı')}</p>
            """
        return f"'{class_name}' hakkında bilgi bulunamadı."
    
    def get_custom_quick_response(self, class_name, custom_query):
        """Özel sorular için JSON'dan hızlı yanıt oluştur"""
        if class_name in self.json_data:
            obj_info = self.json_data[class_name]
            # Soruya göre en uygun bilgiyi seç
            if "ne" in custom_query.lower() or "nedir" in custom_query.lower():
                return f"{obj_info.get('description', 'Bilgi bulunamadı')}"
            elif "özellik" in custom_query.lower():
                return f"{obj_info.get('features', 'Bilgi bulunamadı')}"
            elif "kullanım" in custom_query.lower() or "nasıl" in custom_query.lower():
                return f"{obj_info.get('usage', 'Bilgi bulunamadı')}"
            elif "ilginç" in custom_query.lower():
                return f"{obj_info.get('interesting_facts', 'Bilgi bulunamadı')}"
            else:
                return f"{obj_info.get('description', 'Bilgi bulunamadı')}"
        return f"'{class_name}' hakkında bilgi bulunamadı."
    
    def get_detailed_response(self, query_type, class_name, custom_query=None):
        """LM Studio'dan detaylı yanıt al (arka planda)"""
        try:
            if query_type == 'object_info':
                object_info = self.get_object_info(class_name)
                prompt = f"""Sen bir nesne tanıma asistanısın. Aşağıdaki bilgileri kullanarak 
                '{class_name}' hakkında detaylı bir açıklama yap. Eğlenceli, bilgilendirici ve 
                ilgi çekici ol. Bir öğretmen gibi açıkla.
                
                Nesne bilgileri: {object_info}
                """
            else:
                prompt = f"""Bu bir nesne tanıma sistemidir. Şu nesne tespit edildi: '{class_name}'
                
                Kullanıcı sorusu: {custom_query}
                
                Bu nesne hakkında kullanıcının sorusuna net ve bilgilendirici bir yanıt ver.
                """
            
            # LM Studio'dan yanıt al
            headers = {"Content-Type": "application/json"}
            data = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": "Sen bir nesne tanıma ve bilgilendirme asistanısın."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7,
                "max_tokens": 1000
            }
            
            response = requests.post(
                LM_STUDIO_URL, 
                headers=headers, 
                json=data,
                timeout=3
            )
            
            if response.status_code == 200:
                response_data = response.json()
                assistant_message = response_data['choices'][0]['message']['content']
                self.response_ready.emit(assistant_message, query_type)
                self.connection_error = False
            
        except Exception as e:
            log_message(f"LM Studio detailed response error: {str(e)}", "ERROR")
    
    def get_object_info(self, class_name):
        """Get object info from JSON data"""
        if class_name in self.json_data:
            obj_info = self.json_data[class_name]
            return json.dumps(obj_info)
        return "{}"
    
    def stop(self):
        self.running = False
        self.wait()

# Custom UI Components
class GradientFrame(QFrame):
    """Frame with gradient background"""
    def __init__(self, parent=None, start_color=GRADIENT_START, end_color=GRADIENT_END, 
                direction=Qt.Vertical):
        super().__init__(parent)
        self.start_color = start_color
        self.end_color = end_color
        self.direction = direction
        
    def paintEvent(self, event):
        painter = QPainter(self)
        gradient = QLinearGradient()
        
        if self.direction == Qt.Vertical:
            gradient.setStart(0, 0)
            gradient.setFinalStop(0, self.height())
        else:
            gradient.setStart(0, 0)
            gradient.setFinalStop(self.width(), 0)
        
        gradient.setColorAt(0, self.start_color)
        gradient.setColorAt(1, self.end_color)
        
        painter.fillRect(self.rect(), QBrush(gradient))

class RoundedFrame(QFrame):
    """Frame with rounded corners"""
    def __init__(self, parent=None, radius=10, bg_color=PRIMARY_COLOR):
        super().__init__(parent)
        self.radius = radius
        self.bg_color = bg_color
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Set background color
        painter.setBrush(QBrush(self.bg_color))
        
        # Set border
        painter.setPen(Qt.NoPen)
        
        # Draw rounded rectangle
        painter.drawRoundedRect(self.rect(), self.radius, self.radius)

class AnimatedButton(QPushButton):
    """Button with hover and click animations"""
    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setStyleSheet("""
            QPushButton {
                background-color: #3C6E71;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 8px 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #4D8C8F;
            }
            QPushButton:pressed {
                background-color: #2A5254;
            }
        """)
        
        # Add subtle shadow effect
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setOffset(0, 2)
        shadow.setBlurRadius(5)
        shadow.setColor(QColor(0, 0, 0, 80))
        self.setGraphicsEffect(shadow)

class InfoPanel(QFrame):
    """Information panel with title and content"""
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.setObjectName("infoPanel")
        self.setStyleSheet("""
            #infoPanel {
                background-color: rgba(52, 52, 78, 180);
                border-radius: 8px;
                border: 1px solid rgba(114, 159, 207, 120);
            }
            QLabel {
                color: #E0E0E0;
            }
        """)
        
        layout = QVBoxLayout(self)
        
        # Title
        title_label = QLabel(title)
        title_label.setStyleSheet("""
            font-size: 16px;
            font-weight: bold;
            color: #A0C8E5;
            padding-bottom: 5px;
            border-bottom: 1px solid rgba(114, 159, 207, 120);
        """)
        
        # Content
        self.content = QLabel("Bekliyor...")
        self.content.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.content.setWordWrap(True)
        
        layout.addWidget(title_label)
        layout.addWidget(self.content)
        
    def update_content(self, text):
        self.content.setText(text)

class SettingsDialog(QDialog):
    """Settings dialog for the application"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Ayarlar")
        self.setModal(True)
        self.resize(400, 500)
        
        # Load current settings
        self.settings = load_settings()
        
        # Create UI
        self.init_ui()
        
        # Apply style
        apply_style_sheet(self)
    
    def init_ui(self):
        """Initialize the settings UI"""
        layout = QVBoxLayout(self)
        
        # Camera settings
        camera_group = QGroupBox("Kamera Ayarları")
        camera_layout = QGridLayout()
        
        camera_label = QLabel("Kamera Seçimi:")
        self.camera_combo = QComboBox()
        self.camera_combo.addItems([f"Kamera {i}" for i in get_available_cameras()])
        self.camera_combo.setCurrentIndex(self.settings["camera_index"])
        
        camera_layout.addWidget(camera_label, 0, 0)
        camera_layout.addWidget(self.camera_combo, 0, 1)
        
        camera_group.setLayout(camera_layout)
        layout.addWidget(camera_group)
        
        # Detection settings
        detection_group = QGroupBox("Algılama Ayarları")
        detection_layout = QGridLayout()
        
        conf_label = QLabel("Güven Eşiği:")
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.1, 1.0)
        self.conf_spin.setSingleStep(0.1)
        self.conf_spin.setValue(self.settings["confidence_threshold"])
        
        detection_layout.addWidget(conf_label, 0, 0)
        detection_layout.addWidget(self.conf_spin, 0, 1)
        
        detection_group.setLayout(detection_layout)
        layout.addWidget(detection_group)
        
        # Feature settings
        feature_group = QGroupBox("Özellik Ayarları")
        feature_layout = QGridLayout()
        
        self.sound_check = QCheckBox("Ses Bildirimleri")
        self.sound_check.setChecked(self.settings["sound_enabled"])
        
        self.video_check = QCheckBox("Video Oynatma")
        self.video_check.setChecked(self.settings["video_enabled"])
        
        self.auto_screenshot_check = QCheckBox("Otomatik Ekran Görüntüsü")
        self.auto_screenshot_check.setChecked(self.settings["auto_save_screenshots"])
        
        self.recording_check = QCheckBox("Otomatik Kayıt")
        self.recording_check.setChecked(self.settings["recording_enabled"])
        
        feature_layout.addWidget(self.sound_check, 0, 0)
        feature_layout.addWidget(self.video_check, 0, 1)
        feature_layout.addWidget(self.auto_screenshot_check, 1, 0)
        feature_layout.addWidget(self.recording_check, 1, 1)
        
        feature_group.setLayout(feature_layout)
        layout.addWidget(feature_group)
        
        # Appearance settings
        appearance_group = QGroupBox("Görünüm Ayarları")
        appearance_layout = QGridLayout()
        
        theme_label = QLabel("Tema:")
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Koyu", "Açık"])
        self.theme_combo.setCurrentText("Koyu" if self.settings["dark_mode"] else "Açık")
        
        font_label = QLabel("Yazı Boyutu:")
        self.font_combo = QComboBox()
        self.font_combo.addItems(["Küçük", "Orta", "Büyük"])
        self.font_combo.setCurrentText(self.settings["font_size"].capitalize())
        
        appearance_layout.addWidget(theme_label, 0, 0)
        appearance_layout.addWidget(self.theme_combo, 0, 1)
        appearance_layout.addWidget(font_label, 1, 0)
        appearance_layout.addWidget(self.font_combo, 1, 1)
        
        appearance_group.setLayout(appearance_layout)
        layout.addWidget(appearance_group)
        
        # Language settings
        language_group = QGroupBox("Dil Ayarları")
        language_layout = QGridLayout()
        
        lang_label = QLabel("Dil:")
        self.lang_combo = QComboBox()
        self.lang_combo.addItems(["Türkçe", "English"])
        self.lang_combo.setCurrentText("Türkçe" if self.settings["language"] == "tr" else "English")
        
        language_layout.addWidget(lang_label, 0, 0)
        language_layout.addWidget(self.lang_combo, 0, 1)
        
        language_group.setLayout(language_layout)
        layout.addWidget(language_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        save_button = AnimatedButton("Kaydet")
        save_button.clicked.connect(self.save_settings)
        
        cancel_button = AnimatedButton("İptal")
        cancel_button.clicked.connect(self.reject)
        
        button_layout.addWidget(save_button)
        button_layout.addWidget(cancel_button)
        
        layout.addLayout(button_layout)
    
    def save_settings(self):
        """Save the current settings"""
        self.settings.update({
            "camera_index": self.camera_combo.currentIndex(),
            "confidence_threshold": self.conf_spin.value(),
            "sound_enabled": self.sound_check.isChecked(),
            "video_enabled": self.video_check.isChecked(),
            "dark_mode": self.theme_combo.currentText() == "Koyu",
            "font_size": self.font_combo.currentText().lower(),
            "language": "tr" if self.lang_combo.currentText() == "Türkçe" else "en",
            "auto_save_screenshots": self.auto_screenshot_check.isChecked(),
            "recording_enabled": self.recording_check.isChecked()
        })
        
        save_settings(self.settings)
        self.accept()

class ProfessionalVisionAssistant(QMainWindow):
    """Main application window"""
    def __init__(self):
        super().__init__()
        
        # Load settings
        self.settings = load_settings()
        
        # Setup window
        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}")
        self.resize(1280, 800)
        
        # Initialize data
        self.json_data = load_json_data(JSON_PATH)
        self.video_map = create_video_map(VIDEO_FOLDER)
        self.lm_studio_models = get_available_models()
        self.current_class = None
        self.confidence_threshold = self.settings["confidence_threshold"]
        self.current_model_path = MODEL_PATH
        
        # Initialize recording
        self.recording = False
        self.video_writer = None
        
        # Initialize threads
        self.init_threads()
        
        # Setup UI
        self.init_ui()
        
        # Setup system tray
        self.setup_tray()
        
        # Start detection
        self.detection_thread.start()
        self.video_thread.start()
        self.sound_thread.start()
        self.chat_thread.start()
        
        # Apply style
        apply_style_sheet(self, "dark" if self.settings["dark_mode"] else "light")
        
        # Log startup
        log_message("Application started")
    
    def init_threads(self):
        """Initialize all worker threads"""
        # Detection thread
        self.detection_thread = DetectionThread(
            MODEL_PATH, 
            self.settings["camera_index"], 
            self.confidence_threshold
        )
        self.detection_thread.frame_ready.connect(self.update_detection_frame)
        self.detection_thread.error_occurred.connect(self.handle_detection_error)
        
        # Video player thread
        self.video_thread = VideoThread()
        self.video_thread.frame_ready.connect(self.update_video_frame)
        self.video_thread.error_occurred.connect(self.handle_video_error)
        
        # Sound thread
        self.sound_thread = SoundThread()
        
        # LM Studio chat thread
        self.chat_thread = LMStudioChatThread()
        self.chat_thread.set_json_data(self.json_data)
        self.chat_thread.response_ready.connect(self.update_chat_response)
        self.chat_thread.error_occurred.connect(self.handle_chat_error)
    
    def init_ui(self):
        """Initialize the user interface"""
        # Create central widget with gradient background
        central_widget = GradientFrame(self)
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Left panel (Camera feed and controls)
        left_panel = QVBoxLayout()
        left_panel.setSpacing(8)
        
        # Title
        title_label = QLabel("Nesne Tanıma Sistemi")
        title_label.setStyleSheet("""
            font-size: 24px;
            font-weight: bold;
            color: #A0C8E5;
            padding: 10px;
            background-color: rgba(52, 52, 78, 180);
            border-radius: 8px;
        """)
        title_label.setAlignment(Qt.AlignCenter)
        left_panel.addWidget(title_label)
        
        # Camera feed
        camera_frame = RoundedFrame(radius=8)
        camera_layout = QVBoxLayout(camera_frame)
        
        self.camera_label = QLabel("Kamera başlatılıyor...")
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setStyleSheet("""
            background-color: #2E2E2E;
            color: #C9D6DF;
            border-radius: 8px;
            padding: 10px;
        """)
        self.camera_label.setFixedSize(640, 480)  # Sabit boyut
        self.camera_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)  # Sabit boyut politikası
        camera_layout.addWidget(self.camera_label, 0, Qt.AlignCenter)  # Merkeze hizalama
        
        # Camera controls
        camera_controls = QHBoxLayout()
        
        self.screenshot_button = AnimatedButton("Ekran Görüntüsü")
        self.screenshot_button.clicked.connect(self.take_screenshot)
        
        self.record_button = AnimatedButton("Kayıt Başlat")
        self.record_button.clicked.connect(self.toggle_recording)
        
        camera_controls.addWidget(self.screenshot_button)
        camera_controls.addWidget(self.record_button)
        
        camera_layout.addLayout(camera_controls)
        left_panel.addWidget(camera_frame, 1)
        
        # Controls panel
        controls_frame = RoundedFrame(radius=8)
        controls_layout = QGridLayout(controls_frame)
        
        # Model selection
        model_label = QLabel("YOLO Modeli:")
        model_label.setStyleSheet("color: #E0E0E0;")
        
        model_layout = QHBoxLayout()
        
        self.model_combo = QComboBox()
        self.model_combo.addItems(self.lm_studio_models)
        self.model_combo.setStyleSheet("""
            QComboBox {
                background-color: #3C6E71;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 5px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: none;
            }
        """)
        self.model_combo.currentTextChanged.connect(self.on_model_changed)
        
        self.select_model_button = AnimatedButton("Model Seç")
        self.select_model_button.clicked.connect(self.select_yolo_model)
        
        model_layout.addWidget(self.model_combo)
        model_layout.addWidget(self.select_model_button)
        
        controls_layout.addWidget(model_label, 0, 0)
        controls_layout.addLayout(model_layout, 0, 1)
        
        left_panel.addWidget(controls_frame)
        
        # Right panel (Object info and chat)
        right_panel = QVBoxLayout()
        right_panel.setSpacing(8)
        
        # Video player panel
        video_frame = RoundedFrame(radius=8)
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
        self.video_label.setStyleSheet("""
            background-color: #2E2E2E;
            color: #C9D6DF;
            border-radius: 8px;
            padding: 10px;
            font-style: italic;
        """)
        self.video_label.setFixedSize(320, 240)
        self.video_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        
        # 3D model yükleme butonu
        self.load_model_button = AnimatedButton("3D Model Yükle")
        self.load_model_button.setStyleSheet("""
            QPushButton {
                background-color: #3C6E71;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 8px 12px;
                font-weight: bold;
                margin-top: 10px;
            }
            QPushButton:hover {
                background-color: #4D8C8F;
            }
            QPushButton:disabled {
                background-color: #2A5254;
                color: #888888;
            }
        """)
        self.load_model_button.clicked.connect(self.load_3d_model)
        self.load_model_button.setEnabled(False)  # Başlangıçta devre dışı
        
        video_layout.addWidget(video_title)
        video_layout.addWidget(self.video_label, 0, Qt.AlignCenter)
        video_layout.addWidget(self.load_model_button, 0, Qt.AlignCenter)
        right_panel.addWidget(video_frame)
        
        # Object info panel
        self.object_info_panel = InfoPanel("Nesne Bilgisi")
        right_panel.addWidget(self.object_info_panel)
        
        # Chat panel
        chat_frame = RoundedFrame(radius=8)
        chat_layout = QVBoxLayout(chat_frame)
        
        # Chat history
        self.chat_history = QTextEdit()
        self.chat_history.setReadOnly(True)
        self.chat_history.setStyleSheet("""
            QTextEdit {
                background-color: rgba(52, 52, 78, 180);
                color: #E0E0E0;
                border: none;
                border-radius: 8px;
                padding: 10px;
            }
        """)
        chat_layout.addWidget(self.chat_history)
        
        # Chat input
        input_layout = QHBoxLayout()
        self.chat_input = QLineEdit()
        self.chat_input.setPlaceholderText("Nesne hakkında soru sorun...")
        self.chat_input.setStyleSheet("""
            QLineEdit {
                background-color: #3C6E71;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px;
            }
        """)
        self.chat_input.returnPressed.connect(self.on_chat_input)
        
        send_button = AnimatedButton("Gönder")
        send_button.clicked.connect(self.on_chat_input)
        
        input_layout.addWidget(self.chat_input)
        input_layout.addWidget(send_button)
        chat_layout.addLayout(input_layout)
        
        right_panel.addWidget(chat_frame)
        
        # Add panels to main layout
        main_layout.addLayout(left_panel, 1)
        main_layout.addLayout(right_panel, 1)
    
    def setup_tray(self):
        """Setup system tray icon and menu"""
        self.tray_icon = QSystemTrayIcon(self)
        self.tray_icon.setIcon(QApplication.style().standardIcon(QStyle.SP_ComputerIcon))
        
        # Create tray menu
        tray_menu = QMenu()
        
        show_action = tray_menu.addAction("Göster")
        show_action.triggered.connect(self.show)
        
        hide_action = tray_menu.addAction("Gizle")
        hide_action.triggered.connect(self.hide)
        
        settings_action = tray_menu.addAction("Ayarlar")
        settings_action.triggered.connect(self.show_settings)
        
        tray_menu.addSeparator()
        
        quit_action = tray_menu.addAction("Çıkış")
        quit_action.triggered.connect(self.close)
        
        self.tray_icon.setContextMenu(tray_menu)
        self.tray_icon.show()
    
    def update_detection_frame(self, frame, detections):
        """Update the camera feed with detections"""
        try:
            if frame is not None:
                # Save frame for recording if enabled
                if self.recording and self.video_writer is not None:
                    try:
                        self.video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    except Exception as e:
                        log_message(f"Recording error: {str(e)}", "ERROR")
                
                # Convert to pixmap and display
                pixmap = convert_cv_qt(frame)
                scaled_pixmap = pixmap.scaled(
                    640, 480,  # Sabit boyut
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                self.camera_label.setPixmap(scaled_pixmap)
                self.last_frame = frame
            
            if detections:
                # Get the detection with highest confidence
                best_detection = max(detections, key=lambda x: x['confidence'])
                class_name = best_detection['class_name']
                
                # Update object info
                if class_name != self.current_class:
                    self.current_class = class_name
                    self.update_object_info(class_name)
                    
                    # Play sound if enabled
                    if self.settings["sound_enabled"]:
                        try:
                            self.sound_thread.add_to_queue(class_name)
                        except Exception as e:
                            log_message(f"Sound error: {str(e)}", "ERROR")
                    
                    # Play video if available
                    if class_name in self.video_map:
                        try:
                            video_path = self.video_map[class_name]
                            self.video_thread.set_video(video_path)
                            self.video_label.setText(f"{class_name} görselleştirmesi gösteriliyor...")
                        except Exception as e:
                            log_message(f"Video playback error: {str(e)}", "ERROR")
                            self.video_label.setText(f"{class_name} için 3D model yüklenmemiş.\nYeni model eklemek için '3D Model Yükle' butonunu kullanabilirsiniz.")
                    else:
                        self.video_label.setText(f"{class_name} için 3D model yüklenmemiş.\nYeni model eklemek için '3D Model Yükle' butonunu kullanabilirsiniz.")
                    self.load_model_button.setEnabled(True)
                    
                    # Auto save screenshot if enabled
                    if self.settings["auto_save_screenshots"]:
                        try:
                            self.take_screenshot()
                        except Exception as e:
                            log_message(f"Screenshot error: {str(e)}", "ERROR")
        except Exception as e:
            log_message(f"Frame update error: {str(e)}", "ERROR")
    
    def update_video_frame(self, frame):
        """Update the video display"""
        try:
            if frame is not None:
                pixmap = convert_cv_qt(frame)
                scaled_pixmap = pixmap.scaled(
                    320, 240,  # Video boyutu
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                self.video_label.setPixmap(scaled_pixmap)
        except Exception as e:
            # Video frame hatalarını sessizce logla
            log_message(f"Video frame error: {str(e)}", "ERROR")
            self.video_label.setText("Görselleştirme şu anda kullanılamıyor.")
    
    def update_object_info(self, class_name):
        """Update object information panel"""
        if class_name in self.json_data:
            obj_info = self.json_data[class_name]
            info_text = f"""
            <b>{obj_info.get('name', class_name)}</b>
            
            <p><b>Tanım:</b> {obj_info.get('description', 'Bilgi bulunamadı')}</p>
            
            <p><b>Özellikler:</b> {obj_info.get('features', 'Bilgi bulunamadı')}</p>
            
            <p><b>Kullanım:</b> {obj_info.get('usage', 'Bilgi bulunamadı')}</p>
            
            <p><b>İlginç Bilgi:</b> {obj_info.get('interesting_facts', 'Bilgi bulunamadı')}</p>
            """
            self.object_info_panel.update_content(info_text)
            
            # Ask LM Studio for more info
            self.chat_thread.add_query('object_info', class_name)
    
    def update_chat_response(self, response, query_type):
        """Update chat with LM Studio response"""
        if query_type == 'object_info':
            self.chat_history.append(f"<b>Asistan:</b> {response}")
        else:
            self.chat_history.append(f"<b>Asistan:</b> {response}")
    
    def on_chat_input(self):
        """Handle chat input"""
        if not self.current_class:
            self.chat_history.append("<b>Sistem:</b> Henüz bir nesne tespit edilmedi.")
            return
        
        query = self.chat_input.text().strip()
        if not query:
            return
        
        self.chat_history.append(f"<b>Siz:</b> {query}")
        self.chat_input.clear()
        
        # Send query to LM Studio
        self.chat_thread.add_query('custom', self.current_class, query)
    
    def on_model_changed(self, model_name):
        """Handle model selection change"""
        self.chat_thread.set_model(model_name)
    
    def on_confidence_changed(self, value):
        """Handle confidence threshold change"""
        self.confidence_threshold = value / 100.0
        # Update detection thread with new threshold
        self.detection_thread.conf_threshold = self.confidence_threshold
    
    def take_screenshot(self):
        """Take a screenshot of the current frame"""
        if hasattr(self, 'last_frame'):
            filename = save_screenshot(self.last_frame)
            show_notification(
                "Ekran Görüntüsü",
                f"Ekran görüntüsü kaydedildi: {filename}",
                QSystemTrayIcon.Information
            )
    
    def toggle_recording(self):
        """Toggle video recording"""
        if not self.recording:
            # Start recording
            self.video_writer = create_recording_writer()
            self.recording = True
            self.record_button.setText("Kayıt Durdur")
            show_notification(
                "Video Kaydı",
                "Video kaydı başlatıldı",
                QSystemTrayIcon.Information
            )
        else:
            # Stop recording
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None
            self.recording = False
            self.record_button.setText("Kayıt Başlat")
            show_notification(
                "Video Kaydı",
                "Video kaydı durduruldu",
                QSystemTrayIcon.Information
            )
    
    def show_settings(self):
        """Show settings dialog"""
        try:
            dialog = SettingsDialog(self)
            if dialog.exec():
                # Mevcut ayarları yedekle
                old_settings = self.settings.copy()
                
                # Yeni ayarları yükle
                self.settings = load_settings()
                
                # Kamera değişikliği kontrolü
                if old_settings["camera_index"] != self.settings["camera_index"]:
                    # Mevcut detection thread'i durdur
                    if hasattr(self, 'detection_thread'):
                        self.detection_thread.stop()
                        self.detection_thread.wait()  # Thread'in durmasını bekle
                    
                    # Yeni detection thread başlat
                    self.detection_thread = DetectionThread(
                        self.current_model_path,
                        self.settings["camera_index"],
                        self.settings["confidence_threshold"]
                    )
                    self.detection_thread.frame_ready.connect(self.update_detection_frame)
                    self.detection_thread.error_occurred.connect(self.handle_detection_error)
                    self.detection_thread.start()
                
                # Güven eşiği değişikliği
                if old_settings["confidence_threshold"] != self.settings["confidence_threshold"]:
                    self.confidence_threshold = self.settings["confidence_threshold"]
                    if hasattr(self, 'detection_thread'):
                        self.detection_thread.conf_threshold = self.confidence_threshold
                
                # Tema değişikliği
                if old_settings["dark_mode"] != self.settings["dark_mode"]:
                    apply_style_sheet(self, "dark" if self.settings["dark_mode"] else "light")
                
                # Ses ayarları değişikliği
                if old_settings["sound_enabled"] != self.settings["sound_enabled"]:
                    if not self.settings["sound_enabled"]:
                        self.sound_thread.queue.clear()
                
                # Otomatik kayıt değişikliği
                if old_settings["recording_enabled"] != self.settings["recording_enabled"]:
                    if self.recording and not self.settings["recording_enabled"]:
                        self.toggle_recording()  # Kaydı durdur
                
                QMessageBox.information(self, "Başarılı", "Ayarlar başarıyla güncellendi!")
                
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Ayarlar güncellenirken hata oluştu: {str(e)}")
            log_message(f"Settings update error: {str(e)}", "ERROR")
            # Hata durumunda eski ayarları geri yükle
            self.settings = old_settings
            save_settings(self.settings)
    
    def handle_detection_error(self, error_msg):
        """Handle detection thread errors"""
        log_message(f"Detection error: {error_msg}", "ERROR")
        
        # Kritik hataları göster
        if "camera" in error_msg.lower():
            QMessageBox.warning(self, "Kamera Hatası", 
                              "Kamera bağlantısında sorun oluştu. Lütfen ayarlardan kamera seçimini kontrol edin.")
            self.camera_label.setText("Kamera bağlantı hatası.\nLütfen ayarları kontrol edin.")
            
        elif "model" in error_msg.lower():
            QMessageBox.warning(self, "Model Hatası", 
                              "Model yüklenirken hata oluştu. Lütfen model dosyasını kontrol edin.")
            self.camera_label.setText("Model yükleme hatası.\nLütfen model seçimini kontrol edin.")
            
        else:
            # Diğer hataları sessizce logla
            log_message(f"Silent error: {error_msg}", "ERROR")
            self.camera_label.setText("Bir hata oluştu. Lütfen ayarları kontrol edin.")
    
    def handle_video_error(self, error_msg):
        """Handle video thread errors"""
        log_message(f"Video error: {error_msg}", "ERROR")
        # Video hatalarını sessizce logla ve ekranda gösterme
        self.video_label.setText("Görselleştirme şu anda kullanılamıyor.")
    
    def handle_chat_error(self, error_msg):
        """Handle chat thread errors"""
        log_message(f"Chat error: {error_msg}", "ERROR")
        # Chat hatalarını sessizce logla ve ekranda gösterme
        self.chat_history.append("<b>Sistem:</b> Bağlantı hatası. Lütfen daha sonra tekrar deneyin.")
    
    def closeEvent(self, event):
        """Handle window close event"""
        # Stop recording if active
        if self.recording and self.video_writer is not None:
            self.video_writer.release()
        
        # Stop all threads
        self.detection_thread.stop()
        self.video_thread.stop()
        self.sound_thread.stop()
        self.chat_thread.stop()
        
        # Log shutdown
        log_message("Application closed")
        
        event.accept()

    def load_3d_model(self):
        """3D model yükleme dialog'u"""
        if not self.current_class:
            return
        
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        file_dialog.setNameFilter("3D Model Files (*.obj *.fbx *.glb *.gltf)")
        
        if file_dialog.exec():
            selected_files = file_dialog.selectedFiles()
            if selected_files:
                model_path = selected_files[0]
                # Model klasörünü oluştur
                model_dir = Path(MODELS_FOLDER) / self.current_class
                model_dir.mkdir(parents=True, exist_ok=True)
                
                # Modeli kopyala
                import shutil
                target_path = model_dir / Path(model_path).name
                shutil.copy2(model_path, target_path)
                
                # Video map'i güncelle
                self.video_map[self.current_class] = str(target_path)
                
                # Modeli göster
                self.video_thread.set_video(str(target_path))
                self.video_label.setText(f"{self.current_class} görselleştirmesi yüklendi ve gösteriliyor...")

    def select_yolo_model(self):
        """YOLO model dosyası seçme dialog'u"""
        try:
            file_dialog = QFileDialog()
            file_dialog.setFileMode(QFileDialog.ExistingFile)
            file_dialog.setNameFilter("YOLO Model Files (*.pt)")
            
            if file_dialog.exec():
                selected_files = file_dialog.selectedFiles()
                if selected_files:
                    model_path = selected_files[0]
                    
                    # Model dosyasını kontrol et
                    if not os.path.exists(model_path):
                        QMessageBox.warning(self, "Hata", "Seçilen model dosyası bulunamadı!")
                        return
                    
                    # Modeli models klasörüne kopyala
                    target_path = Path(MODELS_FOLDER) / Path(model_path).name
                    import shutil
                    shutil.copy2(model_path, target_path)
                    
                    # Mevcut detection thread'i durdur
                    if hasattr(self, 'detection_thread'):
                        self.detection_thread.stop()
                        self.detection_thread.wait()  # Thread'in durmasını bekle
                    
                    # Yeni modeli ayarla
                    self.current_model_path = str(target_path)
                    
                    # Yeni detection thread başlat
                    self.detection_thread = DetectionThread(
                        self.current_model_path,
                        self.settings["camera_index"],
                        self.confidence_threshold
                    )
                    self.detection_thread.frame_ready.connect(self.update_detection_frame)
                    self.detection_thread.error_occurred.connect(self.handle_detection_error)
                    self.detection_thread.start()
                    
                    # Bilgilendirme mesajı
                    self.camera_label.setText("Yeni model yüklendi. Kamera başlatılıyor...")
                    QMessageBox.information(self, "Başarılı", "Model başarıyla değiştirildi!")
                    
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Model değiştirme hatası: {str(e)}")
            log_message(f"Model change error: {str(e)}", "ERROR")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ProfessionalVisionAssistant()
    window.show()
    sys.exit(app.exec()) 