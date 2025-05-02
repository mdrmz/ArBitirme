import sys
import os
import time
import json
import cv2
import numpy as np
import requests
from PySide6.QtCore import Qt, QThread, Signal, QSize, QTimer, QPropertyAnimation, QEasingCurve
from PySide6.QtGui import (QImage, QPixmap, QIcon, QFont, QColor, QPalette, 
                          QLinearGradient, QPainter, QBrush)
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, 
                              QVBoxLayout, QHBoxLayout, QPushButton, QComboBox,
                              QTextEdit, QGroupBox, QFrame, QMessageBox, 
                              QProgressBar, QSizePolicy, QSpacerItem)
from ultralytics import YOLO
import qdarkstyle

# --- AYARLAR ---
MODEL_PATH = "yolov8n.pt"
CAMERA_INDEX = 0
VIDEO_FOLDER = "video"
JSON_PATH = "object_info.json"
CONFIDENCE_THRESHOLD = 0.6
SOUND_COOLDOWN = 2.0
LM_STUDIO_BASE_URL = "http://10.52.15.98:40"
LM_STUDIO_URL = f"{LM_STUDIO_BASE_URL}/v1/chat/completions"

# --- ÖZEL WIDGET'LAR ---
class ModernGroupBox(QGroupBox):
    def __init__(self, title, parent=None):
        super().__init__(title, parent)
        self.setStyleSheet("""
            QGroupBox {
                background-color: #2D2D2D;
                border: 2px solid #3D3D3D;
                border-radius: 10px;
                margin-top: 15px;
                padding: 15px;
            }
            QGroupBox::title {
                color: #00B4D8;
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)

class ModernButton(QPushButton):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setStyleSheet("""
            QPushButton {
                background-color: #00B4D8;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 8px 15px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #0096C7;
            }
            QPushButton:pressed {
                background-color: #0077B6;
            }
        """)

class ModernComboBox(QComboBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            QComboBox {
                background-color: #2D2D2D;
                color: white;
                border: 2px solid #3D3D3D;
                border-radius: 5px;
                padding: 5px;
                min-width: 200px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: url(down_arrow.png);
                width: 12px;
                height: 12px;
            }
            QComboBox QAbstractItemView {
                background-color: #2D2D2D;
                color: white;
                selection-background-color: #00B4D8;
            }
        """)

class ModernTextEdit(QTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            QTextEdit {
                background-color: #2D2D2D;
                color: white;
                border: 2px solid #3D3D3D;
                border-radius: 5px;
                padding: 10px;
            }
        """)

class ModernProgressBar(QProgressBar):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            QProgressBar {
                border: 2px solid #3D3D3D;
                border-radius: 5px;
                text-align: center;
                background-color: #2D2D2D;
            }
            QProgressBar::chunk {
                background-color: #00B4D8;
                border-radius: 3px;
            }
        """)

# --- YARDIMCI FONKSİYONLAR ---
def get_available_models():
    try:
        r = requests.get(f"{LM_STUDIO_BASE_URL}/v1/models", timeout=5)
        if r.status_code == 200:
            return [m['id'] for m in r.json()]
    except Exception as e:
        print(f"Model listesi alınamadı: {e}")
    return [
        "deepseek-coder-6.7b-instruct.Q4_K_M.gguf",
        "deepseek-llm-7b-chat.Q4_K_M.gguf",
        "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        "llama-2-7b-chat.Q4_K_M.gguf"
    ]

def load_json_data():
    if os.path.exists(JSON_PATH):
        with open(JSON_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"objects": {}}

def create_video_map():
    video_map = {}
    if os.path.exists(VIDEO_FOLDER):
        for filename in os.listdir(VIDEO_FOLDER):
            if filename.endswith(('.mp4', '.avi', '.mov')):
                class_name = os.path.splitext(filename)[0]
                video_map[class_name] = os.path.join(VIDEO_FOLDER, filename)
    return video_map

def get_fallback_info(class_name, json_data):
    try:
        if 'objects' in json_data and class_name in json_data['objects']:
            obj = json_data['objects'][class_name]
            return (
                f"<h3 style='color: #00B4D8;'>{obj.get('name', class_name).title()}</h3>"
                f"<p>{obj.get('description', '')}</p>"
                f"<h4 style='color: #90E0EF;'>Özellikler:</h4>"
                f"<p>{obj.get('features', 'Bilgi yok')}</p>"
                f"<h4 style='color: #90E0EF;'>Kullanım:</h4>"
                f"<p>{obj.get('usage', 'Bilgi yok')}</p>"
                f"<h4 style='color: #90E0EF;'>İlginç Bilgi:</h4>"
                f"<p>{obj.get('interesting_facts', 'Bilgi yok')}</p>"
            )
    except Exception as e:
        print(f"Fallback bilgi hatası: {e}")
    return f"<p style='color: #FF6B6B;'>{class_name.title()} hakkında hazır bilgi yok.</p>"

# --- THREADLER ---
class DetectionThread(QThread):
    frame_ready = Signal(np.ndarray, list)
    error_occurred = Signal(str)
    
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
            if not cap.isOpened():
                raise Exception("Kamera açılamadı")
            
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    continue
                
                results = model(frame, conf=self.conf_threshold)[0]
                detections = []
                
                for box in results.boxes:
                    cls = int(box.cls)
                    conf = float(box.conf[0])
                    label = model.names[cls]
                    coords = box.xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = coords
                    
                    # Daha şık bir bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 180, 216), 2)
                    cv2.putText(frame, f'{label}: {conf:.2f}', 
                              (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                              0.6, (0, 180, 216), 2)
                    
                    detections.append({
                        'object_id': len(detections)+1,
                        'class_name': label,
                        'confidence': conf,
                        'bounding_box': [x1, y1, x2, y2]
                    })
                
                self.frame_ready.emit(frame, detections)
                time.sleep(0.01)
            
            cap.release()
        except Exception as e:
            self.error_occurred.emit(str(e))
    
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
    
    def set_video(self, path):
        self.video_path = path
    
    def run(self):
        while self.running:
            if self.video_path:
                try:
                    cap = cv2.VideoCapture(self.video_path)
                    if not cap.isOpened():
                        raise Exception(f"Video açılamadı: {self.video_path}")
                    
                    while self.running and self.video_path:
                        ret, frame = cap.read()
                        if not ret:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            continue
                        
                        self.frame_ready.emit(frame)
                        time.sleep(0.01)
                    
                    cap.release()
                except Exception as e:
                    self.error_occurred.emit(str(e))
            else:
                time.sleep(0.01)
    
    def stop(self):
        self.running = False
        self.wait()

class LMStudioThread(QThread):
    response_ready = Signal(str)
    error_occurred = Signal(str)
    
    def __init__(self, object_info, model_name, json_data):
        super().__init__()
        self.object_info = object_info
        self.model_name = model_name
        self.json_data = json_data
    
    def run(self):
        class_name = self.object_info['class_name']
        fallback = get_fallback_info(class_name, self.json_data)
        self.response_ready.emit(fallback + "<br><i style='color: #90E0EF;'>AI yanıtı bekleniyor...</i>")
        
        try:
            headers = {"Content-Type": "application/json"}
            prompt = (
                f"Lütfen aşağıdaki nesne hakkında detaylı bilgi ver:\n"
                f"Nesne: {class_name}\n"
                f"Güven Skoru: {self.object_info['confidence']:.2%}\n"
                "1. Bu nesne nedir?\n2. Genel özellikleri nelerdir?\n"
                "3. Günlük hayatta nasıl kullanılır?\n4. İlginç bilgiler ve tarihçesi\n"
                "Lütfen Türkçe olarak açık ve anlaşılır bir şekilde yanıt ver."
            )
            
            data = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 1000,
                "stream": False
            }
            
            r = requests.post(LM_STUDIO_URL, headers=headers, json=data, timeout=30)
            
            if r.status_code == 200:
                response_data = r.json()
                if 'choices' in response_data and len(response_data['choices']) > 0:
                    msg = response_data['choices'][0]['message']['content']
                    # HTML formatında yanıt
                    formatted_msg = f"<h3 style='color: #00B4D8;'>{class_name.title()}</h3>"
                    formatted_msg += "<div style='color: #E8ECEF;'>" + msg.replace("\n", "<br>") + "</div>"
                    self.response_ready.emit(formatted_msg)
                else:
                    self.error_occurred.emit("AI yanıtı beklenen formatta değil.")
            else:
                self.error_occurred.emit(f"API Hatası: {r.status_code} - {r.text}")
        except Exception as e:
            self.error_occurred.emit(f"AI Hatası: {str(e)}")

# --- ANA UYGULAMA ---
class ModernVisionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Vision Assistant - Modern")
        self.setWindowIcon(QIcon.fromTheme("applications-graphics"))
        self.setGeometry(100, 100, 1800, 1000)
        
        # Veri yükleme
        self.json_data = load_json_data()
        self.video_map = create_video_map()
        
        # Thread'ler
        self.detection_thread = DetectionThread(MODEL_PATH, CAMERA_INDEX, CONFIDENCE_THRESHOLD)
        self.video_thread = VideoThread()
        self.lm_studio_thread = None
        
        # Değişkenler
        self.current_playing_class = None
        self.last_detected_object = None
        self.last_detection_time = 0
        self.detection_cooldown = 5
        
        # UI kurulumu
        self.init_ui()
        self.apply_modern_style()
        
        # Sinyal bağlantıları
        self.detection_thread.frame_ready.connect(self.update_main_frame)
        self.detection_thread.error_occurred.connect(self.show_error)
        self.video_thread.frame_ready.connect(self.update_secondary_frame)
        self.video_thread.error_occurred.connect(self.show_error)
        
        # Thread'leri başlat
        self.detection_thread.start()
        self.video_thread.start()
        
        # Modelleri yükle
        self.load_models()
    
    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main = QHBoxLayout(central)
        main.setSpacing(20)
        main.setContentsMargins(20, 20, 20, 20)
        
        # Sol panel
        left = QVBoxLayout()
        left.setSpacing(15)
        
        # Model seçimi
        model_box = ModernGroupBox("Model Seçimi")
        model_layout = QHBoxLayout()
        self.model_combo = ModernComboBox()
        model_layout.addWidget(self.model_combo)
        self.refresh_btn = ModernButton("Yenile")
        self.refresh_btn.clicked.connect(self.load_models)
        model_layout.addWidget(self.refresh_btn)
        model_box.setLayout(model_layout)
        left.addWidget(model_box)
        
        # Kamera görüntüsü
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("""
            QLabel {
                background-color: #1A1A1A;
                border: 2px solid #3D3D3D;
                border-radius: 10px;
                padding: 10px;
            }
        """)
        left.addWidget(self.video_label, 1)
        
        # Tespit edilen nesneler
        objects_box = ModernGroupBox("Tespit Edilen Nesneler")
        objects_layout = QVBoxLayout()
        self.objects_text = ModernTextEdit()
        self.objects_text.setMaximumHeight(120)
        objects_layout.addWidget(self.objects_text)
        objects_box.setLayout(objects_layout)
        left.addWidget(objects_box)
        
        main.addLayout(left, 1)
        
        # Orta ayraç
        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setStyleSheet("background-color: #3D3D3D;")
        main.addWidget(sep)
        
        # Sağ panel
        right = QVBoxLayout()
        right.setSpacing(15)
        
        # Video gösterimi
        self.secondary_video_label = QLabel()
        self.secondary_video_label.setAlignment(Qt.AlignCenter)
        self.secondary_video_label.setMinimumSize(640, 480)
        self.secondary_video_label.setStyleSheet("""
            QLabel {
                background-color: #1A1A1A;
                border: 2px solid #3D3D3D;
                border-radius: 10px;
                padding: 10px;
            }
        """)
        right.addWidget(self.secondary_video_label, 1)
        
        # AI açıklama
        ai_box = ModernGroupBox("AI Açıklaması")
        ai_layout = QVBoxLayout()
        self.response_text = ModernTextEdit()
        ai_layout.addWidget(self.response_text)
        ai_box.setLayout(ai_layout)
        right.addWidget(ai_box)
        
        # Yükleniyor barı
        self.progress = ModernProgressBar()
        self.progress.setMaximum(0)
        self.progress.setVisible(False)
        right.addWidget(self.progress)
        
        main.addLayout(right, 1)
    
    def apply_modern_style(self):
        self.setStyleSheet(qdarkstyle.load_stylesheet())
        self.setFont(QFont("Segoe UI", 11))
        
        # Ana pencere arka plan rengi
        palette = self.palette()
        palette.setColor(QPalette.Window, QColor("#1A1A1A"))
        self.setPalette(palette)
    
    def load_models(self):
        self.model_combo.clear()
        self.model_combo.addItem("Yükleniyor...")
        QApplication.processEvents()
        
        models = get_available_models()
        self.model_combo.clear()
        
        for m in models:
            self.model_combo.addItem(m)
        
        if models:
            self.model_combo.setCurrentText(models[0])
    
    def update_main_frame(self, frame, detections):
        self.display_image(frame, self.video_label)
        self.update_objects_list(detections)
        self.process_detections(detections)
    
    def update_objects_list(self, detections):
        text = ""
        for d in detections:
            text += f"<b style='color: #00B4D8;'>{d['object_id']}:</b> "
            text += f"<span style='color: #E8ECEF;'>{d['class_name']}</span> "
            text += f"<span style='color: #90E0EF;'>({d['confidence']:.2%})</span><br>"
        self.objects_text.setHtml(text)
    
    def update_secondary_frame(self, frame):
        self.display_image(frame, self.secondary_video_label)
    
    def display_image(self, frame, label):
        if frame is not None:
            h, w = frame.shape[:2]
            label_size = label.size()
            aspect = w / h
            
            if label_size.width() / label_size.height() > aspect:
                new_h = label_size.height()
                new_w = int(new_h * aspect)
            else:
                new_w = label_size.width()
                new_h = int(new_w / aspect)
            
            img = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            qimg = QImage(rgb.data, new_w, new_h, 3 * new_w, QImage.Format_RGB888)
            label.setPixmap(QPixmap.fromImage(qimg))
    
    def process_detections(self, detections):
        if not detections:
            self.stop_secondary_video()
            return
        
        best = max(detections, key=lambda x: x['confidence'])
        now = time.time()
        
        if best['class_name'] in self.video_map:
            self.play_video_based_on_class(best['class_name'])
        
        if (self.last_detected_object != best['class_name'] or 
            now - self.last_detection_time > self.detection_cooldown):
            
            self.last_detected_object = best['class_name']
            self.last_detection_time = now
            
            self.response_text.setHtml(
                f"<h3 style='color: #00B4D8;'>Tespit Edilen Nesne:</h3>"
                f"<p style='color: #E8ECEF;'>{best['class_name']}</p>"
                f"<i style='color: #90E0EF;'>Analiz ediliyor...</i>"
            )
            
            self.progress.setVisible(True)
            
            if self.lm_studio_thread and self.lm_studio_thread.isRunning():
                self.lm_studio_thread.terminate()
            
            self.lm_studio_thread = LMStudioThread(
                best, 
                self.model_combo.currentText(), 
                self.json_data
            )
            self.lm_studio_thread.response_ready.connect(self.update_lm_studio_response)
            self.lm_studio_thread.error_occurred.connect(self.show_error)
            self.lm_studio_thread.start()
    
    def update_lm_studio_response(self, response):
        self.response_text.setHtml(response)
        self.progress.setVisible(False)
    
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
    
    def show_error(self, msg):
        self.progress.setVisible(False)
        QMessageBox.critical(self, "Hata", str(msg))
    
    def closeEvent(self, event):
        self.detection_thread.stop()
        self.video_thread.stop()
        if self.lm_studio_thread:
            self.lm_studio_thread.terminate()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ModernVisionApp()
    window.show()
    sys.exit(app.exec()) 