import sys
import cv2
import numpy as np
import requests
import json
from PySide6.QtCore import QTimer, Qt, QSize, QThread, Signal, QObject
from PySide6.QtGui import QImage, QPixmap, QFont, QPalette, QColor, QIcon
from PySide6.QtWidgets import (QApplication, QWidget, QLabel, QVBoxLayout,
                             QHBoxLayout, QFrame, QSizePolicy, QGroupBox,
                             QLineEdit, QPushButton, QTextEdit, QMessageBox,
                             QMainWindow, QComboBox)
import time
import os
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
LM_STUDIO_BASE_URL = "http://10.52.15.98:40"
LM_STUDIO_URL = f"{LM_STUDIO_BASE_URL}/v1/chat/completions"

# Available models
AVAILABLE_MODELS = [
    "deepseek-coder-6.7b-instruct.Q4_K_M.gguf",
    "deepseek-llm-7b-chat.Q4_K_M.gguf",
    "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    "llama-2-7b-chat.Q4_K_M.gguf"
]

def get_available_models():
    try:
        print(f"Modeller alınıyor: {LM_STUDIO_BASE_URL}/v1/models")
        response = requests.get(f"{LM_STUDIO_BASE_URL}/v1/models", timeout=5)
        print(f"Model listesi yanıtı: {response.status_code}")
        
        if response.status_code == 200:
            models = response.json()
            print(f"Bulunan modeller: {models}")
            return [model['id'] for model in models]
        print(f"Model listesi alınamadı. Varsayılan modeller kullanılacak.")
        return AVAILABLE_MODELS
    except Exception as e:
        print(f"Model listesi hatası: {e}")
        return AVAILABLE_MODELS

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
            self.model = YOLO(self.model_path)
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                raise Exception("Failed to open camera")
            
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    continue
                
                results = self.model(frame, conf=self.conf_threshold)[0]
                detections = []
                
                for box in results.boxes:
                    cls = int(box.cls)
                    conf = float(box.conf[0])
                    label = self.model.names[cls]
                    coords = box.xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = coords
                    
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
                        time.sleep(0.01)
                    
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

class LMStudioThread(QThread):
    response_ready = Signal(str)
    error_occurred = Signal(str)
    
    def __init__(self, object_info, model_name, json_data):
        super().__init__()
        self.object_info = object_info
        self.model_name = model_name
        self.json_data = json_data
        
    def get_fallback_info(self, class_name):
        try:
            if 'objects' in self.json_data and class_name in self.json_data['objects']:
                obj_info = self.json_data['objects'][class_name]
                return f"""
                {obj_info['name']} hakkında bilgi:
                
                1. Bu nesne nedir?
                {obj_info['description']}
                
                2. Genel özellikleri:
                - {obj_info.get('features', 'Bilgi bulunamadı')}
                
                3. Günlük hayatta kullanımı:
                - {obj_info.get('usage', 'Bilgi bulunamadı')}
                
                4. İlginç bilgiler:
                - {obj_info.get('interesting_facts', 'Bilgi bulunamadı')}
                """
            return f"Bu {class_name} hakkında hazır bilgi bulunamadı."
        except Exception as e:
            print(f"Fallback bilgi hatası: {e}")
            return f"Bu {class_name} hakkında bilgi yüklenemedi."

    def run(self):
        try:
            print(f"LM Studio bağlantısı başlatılıyor... Model: {self.model_name}")
            
            # Önce fallback bilgiyi göster
            fallback_info = self.get_fallback_info(self.object_info['class_name'])
            self.response_ready.emit(fallback_info + "\n\nLM Studio'dan detaylı bilgi bekleniyor...")
            
            # API bağlantısını test et
            test_url = "http://10.52.15.98:40/v1/models"
            print(f"API test isteği gönderiliyor: {test_url}")
            test_response = requests.get(test_url, timeout=5)
            print(f"API test yanıtı: {test_response.status_code}")
            
            if test_response.status_code != 200:
                raise Exception(f"LM Studio API'sine bağlanılamadı. Durum kodu: {test_response.status_code}")
            
            headers = {
                "Content-Type": "application/json"
            }
            
            prompt = f"""
            Lütfen aşağıdaki nesne hakkında detaylı bilgi ver:
            
            Nesne: {self.object_info['class_name']}
            Güven Skoru: {self.object_info['confidence']:.2%}
            
            Şu başlıklar altında detaylı bir açıklama yap:
            1. Bu nesne nedir?
            2. Genel özellikleri nelerdir?
            3. Günlük hayatta nasıl kullanılır?
            4. İlginç bilgiler ve tarihçesi
            
            Lütfen Türkçe olarak açık ve anlaşılır bir şekilde yanıt ver.
            """
            
            print(f"API isteği gönderiliyor... Prompt: {prompt[:100]}...")
            
            data = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 1000,
                "stream": False
            }
            
            print(f"API isteği verisi: {data}")
            
            response = requests.post(
                "http://10.52.15.98:40/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            
            print(f"API yanıtı alındı. Durum kodu: {response.status_code}")
            print(f"API yanıt içeriği: {response.text[:200]}...")
            
            if response.status_code == 200:
                try:
                    response_data = response.json()
                    print(f"API yanıtı JSON: {response_data}")
                    
                    if 'choices' in response_data and len(response_data['choices']) > 0:
                        assistant_message = response_data['choices'][0]['message']['content']
                        print(f"İşlenmiş yanıt: {assistant_message[:100]}...")
                        self.response_ready.emit(assistant_message)
                    else:
                        error_msg = f"API yanıtı beklenen formatta değil. Yanıt: {response_data}"
                        print(error_msg)
                        self.error_occurred.emit(error_msg)
                except json.JSONDecodeError as e:
                    error_msg = f"API yanıtı JSON formatında değil. Hata: {str(e)}"
                    print(error_msg)
                    self.error_occurred.emit(error_msg)
            else:
                error_msg = f"API Hatası: {response.status_code} - {response.text}"
                print(error_msg)
                self.error_occurred.emit(error_msg)
                
        except requests.exceptions.Timeout:
            error_msg = "API yanıt vermedi (30 saniye zaman aşımı)"
            print(error_msg)
            self.error_occurred.emit(error_msg)
        except requests.exceptions.ConnectionError as e:
            error_msg = f"LM Studio sunucusuna bağlanılamadı: {str(e)}"
            print(error_msg)
            self.error_occurred.emit(error_msg)
        except Exception as e:
            error_msg = f"LM Studio Hatası: {str(e)}"
            print(error_msg)
            self.error_occurred.emit(error_msg)

    def handle_lm_studio_error(self, error_msg):
        print(f"LM Studio hatası işleniyor: {error_msg}")
        QMessageBox.critical(self, "LM Studio Hatası", error_msg)
        self.response_text.setText(f"Hata: {error_msg}\n\nLütfen LM Studio'nun çalışır durumda olduğunu ve seçili modelin yüklü olduğunu kontrol edin.")

    def process_detections(self, detections):
        if not detections:
            self.stop_secondary_video()
            return
        
        best_detection = max(detections, key=lambda x: x['confidence'])
        current_time = time.time()
        
        # Video ve ses işlemleri
        if best_detection['class_name'] in self.video_map:
            self.play_video_based_on_class(best_detection['class_name'])
        
        self.sound_thread.play_sound(best_detection['class_name'])
        
        # LM Studio isteği için kontrol
        should_request = False
        
        # Yeni nesne tespit edildi mi?
        if self.last_detected_object != best_detection['class_name']:
            should_request = True
            print(f"Yeni nesne tespit edildi: {best_detection['class_name']}")
        
        # Son istekten bu yana yeterli süre geçti mi?
        elif current_time - self.last_detection_time > self.detection_cooldown:
            should_request = True
            print(f"Bekleme süresi doldu: {best_detection['class_name']}")
        
        # LM Studio isteği gönder
        if should_request and (self.lm_studio_thread is None or not self.lm_studio_thread.isRunning()):
            self.last_detected_object = best_detection['class_name']
            self.last_detection_time = current_time
            
            self.response_text.setText(f"Tespit edilen nesne: {best_detection['class_name']}\nAnaliz ediliyor...")
            self.lm_studio_thread = LMStudioThread(best_detection, self.model_combo.currentText(), self.json_data)
            self.lm_studio_thread.response_ready.connect(self.update_lm_studio_response)
            self.lm_studio_thread.error_occurred.connect(self.handle_lm_studio_error)
            self.lm_studio_thread.start()

    def update_lm_studio_response(self, response):
        self.response_text.setText(response)
        self.lm_studio_thread = None

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

    def handle_lm_studio_error(self, error_msg):
        self.handle_lm_studio_error(error_msg)

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

    def refresh_models(self):
        try:
            models = get_available_models()
            self.model_combo.clear()
            for model in models:
                self.model_combo.addItem(model)
            if models:
                self.model_combo.setCurrentText(models[0])
            QMessageBox.information(self, "Başarılı", "Modeller başarıyla yenilendi!")
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Model yenileme hatası: {str(e)}")

    def load_models(self):
        try:
            print("Modeller yükleniyor...")
            models = get_available_models()
            self.model_combo.clear()
            for model in models:
                self.model_combo.addItem(model)
            if models:
                self.model_combo.setCurrentText(models[0])
                print(f"Seçilen model: {models[0]}")
        except Exception as e:
            print(f"Model yükleme hatası: {e}")
            QMessageBox.warning(self, "Uyarı", "Modeller yüklenemedi. Varsayılan modeller kullanılacak.")

    def closeEvent(self, event):
        # Stop all threads
        self.detection_thread.stop()
        self.video_thread.stop()
        self.sound_thread.stop()
        if self.lm_studio_thread:
            self.lm_studio_thread.wait()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ObjectDetectionApp()
    window.show()
    sys.exit(app.exec()) 