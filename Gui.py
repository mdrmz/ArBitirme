import sys
import cv2
import numpy as np
from PySide6.QtCore import QTimer, Qt, QSize
from PySide6.QtGui import QImage, QPixmap, QFont
from PySide6.QtWidgets import (QApplication, QWidget, QLabel, QVBoxLayout,
                             QHBoxLayout, QFrame, QSizePolicy, QGroupBox)
from ultralytics import YOLO
import os
import json
from Sound_Project import Sound

# --- Konfigürasyon ---
MODEL_PATH = "yolov8n.pt"
CAMERA_INDEX = 0
TIMER_INTERVAL_MS = 30
VIDEO_FOLDER = "video"
JSON_PATH = "object_info.json"
CONFIDENCE_THRESHOLD = 0.6
# ---------------------

# Model yükleme
try:
    model = YOLO(MODEL_PATH)
    print(f"YOLO modeli '{MODEL_PATH}' başarıyla yüklendi.")
    if not hasattr(model, 'names') or not model.names:
        raise ValueError("Model isimleri (names) yüklenemedi veya boş.")
    print("Model sınıfları:", model.names)
except Exception as e:
    print(f"HATA: YOLO modeli yüklenemedi: {e}")
    sys.exit(1)

class ObjectDetectionApp(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Object Detection GUI")
        self.setGeometry(100, 100, 1200, 700)
        self.setStyleSheet("background-color: #1C2526;")  # Koyu tema arka planı

        self.videoCapture = cv2.VideoCapture(CAMERA_INDEX)
        if not self.videoCapture.isOpened():
            print(f"HATA: Kamera {CAMERA_INDEX} açılamadı.")
            sys.exit(1)

        self.frame_width = int(self.videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Kamera çözünürlüğü: {self.frame_width}x{self.frame_height}")

        self.secondary_video_capture = None
        self.secondary_timer = None
        self.current_playing_class = None
        self.object_detected_in_frame = False

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(TIMER_INTERVAL_MS)

        self.json_data = self.load_json_data()
        self.init_ui()
        self.create_video_map()

    def load_json_data(self):
        """JSON dosyasını okur ve veriyi döndürür."""
        try:
            if not os.path.exists(JSON_PATH):
                print(f"Uyarı: JSON dosyası '{JSON_PATH}' bulunamadı.")
                return {}
            with open(JSON_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if "objects" not in data:
                    raise ValueError("JSON dosyasında 'objects' anahtarı bulunamadı.")
                print("JSON verisi başarıyla yüklendi:", list(data["objects"].keys()))
                return data["objects"]
        except Exception as e:
            print(f"Hata: JSON dosyası yüklenemedi: {e}")
            return {}

    def init_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # --- Sol Panel ---
        left_panel = QVBoxLayout()
        left_panel.setSpacing(8)

        # Başlık
        left_title = QLabel("Kamera Görüntüsü")
        left_title.setStyleSheet("color: #C9D6DF; font-size: 16px; font-weight: bold; padding: 5px;")
        left_title.setAlignment(Qt.AlignCenter)
        left_panel.addWidget(left_title)

        # Ana video ekranı
        self.video_label = QLabel("Kamera bekleniyor...", self)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("""
            background-color: #2E2E2E;
            color: #C9D6DF;
            border: 2px solid #52616B;
            border-radius: 8px;
        """)
        self.video_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        left_panel.addWidget(self.video_label, 1)

        # Nesne özellikleri çerçevesi
        features_frame = QGroupBox("Nesne Özellikleri")
        features_frame.setStyleSheet("""
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
        features_layout = QVBoxLayout()
        self.features_label = QLabel("Nesne bilgileri bekleniyor...")
        self.features_label.setStyleSheet("""
            color: #E8ECEF;
            background-color: #2E2E2E;
            padding: 10px;
            border-radius: 4px;
        """)
        self.features_label.setWordWrap(True)
        features_layout.addWidget(self.features_label)
        features_frame.setLayout(features_layout)
        left_panel.addWidget(features_frame)

        # --- Sağ Panel ---
        right_panel = QVBoxLayout()
        right_panel.setSpacing(8)

        # Başlık
        right_title = QLabel("Algılanan Nesne Bilgileri")
        right_title.setStyleSheet("color: #C9D6DF; font-size: 16px; font-weight: bold; padding: 5px;")
        right_title.setAlignment(Qt.AlignCenter)
        right_panel.addWidget(right_title)

        # İkincil video ekranı
        self.secondary_video_label = QLabel("Algılanan nesne videosu burada gösterilecek", self)
        self.secondary_video_label.setAlignment(Qt.AlignCenter)
        self.secondary_video_label.setStyleSheet("""
            background-color: #2E2E2E;
            color: #C9D6DF;
            border: 2px solid #52616B;
            border-radius: 8px;
        """)
        self.secondary_video_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        right_panel.addWidget(self.secondary_video_label, 1)

        # JSON bilgileri çerçevesi
        json_frame = QGroupBox("Nesne Detayları")
        json_frame.setStyleSheet("""
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
        json_layout = QHBoxLayout()
        json_layout.setSpacing(10)

        # İkon alanı
        self.json_icon_label = QLabel()
        self.json_icon_label.setFixedSize(48, 48)
        self.json_icon_label.setStyleSheet("background-color: transparent;")
        self.json_icon_label.setAlignment(Qt.AlignCenter)
        json_layout.addWidget(self.json_icon_label)

        # JSON bilgileri
        self.json_info_label = QLabel("Algılanan nesne bilgileri burada gösterilecek.")
        self.json_info_label.setStyleSheet("""
            color: #E8ECEF;
            background-color: #2E2E2E;
            padding: 10px;
            border-radius: 4px;
        """)
        self.json_info_label.setWordWrap(True)
        json_layout.addWidget(self.json_info_label, 1)
        json_frame.setLayout(json_layout)
        right_panel.addWidget(json_frame)

        # Panelleri ekle
        main_layout.addLayout(left_panel, 1)
        separator = QFrame()
        separator.setFrameShape(QFrame.VLine)
        separator.setFrameShadow(QFrame.Sunken)
        separator.setStyleSheet("color: #52616B;")
        main_layout.addWidget(separator)
        main_layout.addLayout(right_panel, 1)
        self.setLayout(main_layout)

    def create_video_map(self):
        """VIDEO_FOLDER içindeki videoları tarayarak video_map oluşturur."""
        self.video_map = {}
        if not os.path.isdir(VIDEO_FOLDER):
            print(f"Uyarı: Video klasörü '{VIDEO_FOLDER}' bulunamadı.")
            return

        print(f"Video klasörü '{VIDEO_FOLDER}' taranıyor...")
        valid_extensions = ('.mp4', '.avi', '.mov', '.mkv')
        for filename in os.listdir(VIDEO_FOLDER):
            if filename.lower().endswith(valid_extensions):
                class_name = os.path.splitext(filename)[0].lower()
                video_path = os.path.join(VIDEO_FOLDER, filename)
                self.video_map[class_name] = video_path
                print(f"  -> Bulunan video: '{class_name}' -> '{video_path}'")
        print("Video haritası oluşturuldu:", self.video_map)

    def update_frame(self):
        """Ana kamera görüntüsünü günceller ve nesne tespiti yapar."""
        ret, frame = self.videoCapture.read()
        if not ret:
            print("Hata: Kamera karesi okunamadı.")
            return

        processed_frame = frame.copy()
        results = model(processed_frame, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]
        self.object_detected_in_frame = len(results.boxes) > 0

        processed_frame = self.process_detections(processed_frame, results)
        self.display_image(processed_frame, self.video_label)

        if not self.object_detected_in_frame:
            self.stop_secondary_video()
            self.features_label.setText("Nesne bilgileri bekleniyor...")
            self.json_info_label.setText("Algılanan nesne bilgileri burada gösterilecek.")
            self.json_icon_label.clear()

    def process_detections(self, frame, results):
        """Algılanan nesneleri işler, kutu çizer, bilgileri gösterir ve videoyu tetikler."""
        object_info_text = ""
        json_info_text = ""


        if not results.boxes:
            self.current_playing_class = None
            return frame

        first_box = results.boxes[0]
        cls = int(first_box.cls)
        conf = float(first_box.conf[0])
        label = model.names[cls] if cls < len(model.names) else f"Unknown ({cls})"
        # burada ise tespit edilen nesnenin isimi okuyacak sürekli
        Sound.ses(label)
        box_coords = first_box.xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = box_coords
        width = x2 - x1
        height = y2 - y1
        object_info_text = (
            f"<b>Nesne:</b> {label}<br>"
            f"<b>Güven:</b> {conf:.2f}<br>"
            f"<b>Konum (Sol-Üst):</b> ({x1}, {y1})<br>"
            f"<b>Boyut (G x Y):</b> {width} x {height}"
        )

        label_lower = label.lower()
        if label_lower in self.json_data:
            json_info = self.json_data[label_lower]
            json_info_text = (
                f"<b>Ad:</b> {json_info.get('display_name', label)}<br>"
                f"<b>Açıklama:</b> {json_info.get('description', 'Bilgi yok')}<br>"
                f"<b>Ek Bilgi:</b> {json_info.get('extra_info', 'Bilgi yok')}"
            )
            # İkon yükleme
            icon_path = json_info.get('icon_path')
            if icon_path and os.path.exists(icon_path):
                pixmap = QPixmap(icon_path).scaled(48, 48, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.json_icon_label.setPixmap(pixmap)
            else:
                self.json_icon_label.clear()
        else:
            json_info_text = f"'{label}' için JSON verisi bulunamadı."
            self.json_icon_label.clear()

        self.play_video_based_on_class(label)
        self.features_label.setText(object_info_text)
        self.json_info_label.setText(json_info_text)

        for box in results.boxes:
            cls_i = int(box.cls)
            label_i = model.names[cls_i] if cls_i < len(model.names) else f"Unknown ({cls_i})"
            coords_i = box.xyxy[0].cpu().numpy().astype(int)
            x1_i, y1_i, x2_i, y2_i = coords_i
            conf_i = float(box.conf[0])

            cv2.rectangle(frame, (x1_i, y1_i), (x2_i, y2_i), (0, 255, 0), 2)
            text = f"{label_i}: {conf_i:.2f}"
            (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (x1_i, y1_i - text_height - baseline), (x1_i + text_width, y1_i), (0, 255, 0), -1)
            cv2.putText(frame, text, (x1_i, y1_i - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        return frame

    def play_video_based_on_class(self, object_class):
        """Algılanan nesne sınıfına göre ilgili videoyu sağ panelde oynatır."""
        object_class_lower = object_class.lower()

        if self.current_playing_class == object_class_lower:
            return

        video_path = self.video_map.get(object_class_lower)

        if video_path and os.path.exists(video_path):
            self.stop_secondary_video(clear_label=False)
            print(f"'{object_class}' algılandı. Video oynatılıyor: {video_path}")
            self.current_playing_class = object_class_lower
            self.play_video(video_path)
        else:
            if self.current_playing_class is not None:
                self.stop_secondary_video()
                self.clear_label(self.secondary_video_label, f"'{object_class}' için video bulunamadı.")
            self.current_playing_class = None

    def play_video(self, video_path):
        """Verilen video dosyasını ikincil video panelinde oynatmaya başlar."""
        try:
            self.secondary_video_capture = cv2.VideoCapture(video_path)
            if not self.secondary_video_capture.isOpened():
                raise IOError(f"Video dosyası açılamadı: {video_path}")

            if self.secondary_timer is None:
                self.secondary_timer = QTimer(self)
                self.secondary_timer.timeout.connect(self.update_secondary_frame)

            fps = self.secondary_video_capture.get(cv2.CAP_PROP_FPS)
            interval = int(1000 / fps) if fps > 0 else TIMER_INTERVAL_MS
            self.secondary_timer.start(interval)

        except Exception as e:
            print(f"Hata: Video oynatma başlatılamadı: {e}")
            self.stop_secondary_video()

    def update_secondary_frame(self):
        """İkincil video panelindeki kareyi günceller."""
        if self.secondary_video_capture is not None and self.secondary_video_capture.isOpened():
            ret, frame = self.secondary_video_capture.read()

            if ret:
                self.display_image(frame, self.secondary_video_label)
            else:
                print(f"'{self.current_playing_class}' videosu bitti.")
                self.stop_secondary_video()
                self.current_playing_class = None
        else:
            self.stop_secondary_video()

    def stop_secondary_video(self, clear_label=True):
        """İkincil video oynatmayı durdurur ve kaynakları serbest bırakır."""
        if self.secondary_timer is not None and self.secondary_timer.isActive():
            self.secondary_timer.stop()

        if self.secondary_video_capture is not None:
            self.secondary_video_capture.release()
            self.secondary_video_capture = None

        if clear_label:
            self.clear_label(self.secondary_video_label, "Algılanan nesne videosu burada gösterilecek")
            self.current_playing_class = None

    def display_image(self, frame, label_widget):
        """Verilen OpenCV karesini belirtilen QLabel widget'ında gösterir."""
        try:
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            scaled_pixmap = pixmap.scaled(
                label_widget.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            label_widget.setPixmap(scaled_pixmap)
        except Exception as e:
            print(f"Hata: Görüntü gösterilirken hata oluştu: {e}")

    def clear_label(self, label_widget, text=""):
        """QLabel içeriğini temizler ve isteğe bağlı bir metin gösterir."""
        label_widget.setText(text)

    def resizeEvent(self, event):
        """Pencere yeniden boyutlandırıldığında çağrılır."""
        super().resizeEvent(event)

    def closeEvent(self, event):
        """Uygulama kapatıldığında kaynakları serbest bırakır."""
        print("Uygulama kapatılıyor, kaynaklar serbest bırakılıyor...")
        self.timer.stop()
        self.stop_secondary_video()

        if self.videoCapture.isOpened():
            self.videoCapture.release()
            print("Ana kamera serbest bırakıldı.")

        event.accept()

if __name__ == '__main__':
    if not os.path.isdir(VIDEO_FOLDER):
        print(f"Uyarı: '{VIDEO_FOLDER}' adında bir klasör bulunamadı.")
        print("Lütfen algılanan nesneler için (örn: person.mp4, car.mp4) videoları içeren bu klasörü oluşturun.")

    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # Modern bir stil için
    window = ObjectDetectionApp()
    window.show()
    sys.exit(app.exec())