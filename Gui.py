import sys
import cv2
import numpy as np
from PySide6.QtCore import QTimer, Qt, QSize
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QApplication, QWidget, QLabel, QVBoxLayout,
                             QPushButton, QTextEdit, QHBoxLayout, QFrame, QSizePolicy) # QSizePolicy eklendi

from ultralytics import YOLO
import os # Video dosyalarının varlığını kontrol etmek için eklendi

# --- Konfigürasyon ---
MODEL_PATH = "yolov8n.pt" # Kullanılacak YOLO modelinin yolu veya adı
CAMERA_INDEX = 0 # Kullanılacak kamera indeksi (genellikle 0)
TIMER_INTERVAL_MS = 30 # Kamera kare hızı (milisaniye cinsinden)
VIDEO_FOLDER = "video" # Algılanan nesneler için videoların bulunduğu klasör
CONFIDENCE_THRESHOLD = 0.5 # Algılama için minimum güven skoru
# ---------------------

# Model yükleme (program başında bir kez yapılır)
try:
    model = YOLO(MODEL_PATH)
    print(f"YOLO modeli '{MODEL_PATH}' başarıyla yüklendi.")
    # Model sınıflarını kontrol et
    if not hasattr(model, 'names') or not model.names:
        raise ValueError("Model isimleri (names) yüklenemedi veya boş.")
    print("Model sınıfları:", model.names)
except Exception as e:
    print(f"HATA: YOLO modeli yüklenemedi: {e}")
    sys.exit(1) # Model yüklenemezse programdan çık

class ObjectDetectionApp(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Object Detection GUI")
        self.setGeometry(100, 100, 1200, 700) # Pencere boyutu ayarlandı

        self.videoCapture = cv2.VideoCapture(CAMERA_INDEX) # Kamera aç
        if not self.videoCapture.isOpened():
            print(f"HATA: Kamera {CAMERA_INDEX} açılamadı.")
            # Alternatif kameraları dene veya hata mesajı göster
            # self.videoCapture = cv2.VideoCapture(1) # Alternatif kamera
            # if not self.videoCapture.isOpened(): ...
            sys.exit(1) # Kamera açılamazsa çık

        # Video frame boyutlarını al (varsa)
        self.frame_width = int(self.videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Kamera çözünürlüğü: {self.frame_width}x{self.frame_height}")

        self.secondary_video_capture = None # İkincil video için capture nesnesi
        self.secondary_timer = None # İkincil video için timer
        self.current_playing_class = None # Hangi sınıfın videosunun oynatıldığını takip eder
        self.object_detected_in_frame = False # Mevcut karede nesne algılanıp algılanmadığını tutar

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(TIMER_INTERVAL_MS)

        self.init_ui()
        self.create_video_map() # Video haritasını oluştur

    def init_ui(self):
        # Ana layout
        main_layout = QHBoxLayout(self)

        # --- Sol Panel (Ana Video ve Bilgiler) ---
        left_panel = QVBoxLayout()

        # Ana video ekranı (QLabel)
        self.video_label = QLabel("Kamera bekleniyor...", self)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: #2E2E2E; color: white; border: 1px solid gray;")
        # QSizePolicy.Ignored: Widget'ın layout içinde mümkün olduğunca genişlemesini sağlar
        self.video_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        left_panel.addWidget(self.video_label, 1) # Stretch faktörü 1 verildi

        # Nesne Özellikleri Paneli (QTextEdit)
        self.features_label = QTextEdit(self)
        self.features_label.setReadOnly(True)
        self.features_label.setMaximumHeight(100) # Yüksekliği sınırlayalım
        self.features_label.setStyleSheet("background-color: #F0F0F0; border: 1px solid gray;")
        left_panel.addWidget(self.features_label) # Stretch 0 (varsayılan)

        # Buton (artık otomatik çalıştığı için isteğe bağlı)
        # self.detect_button = QPushButton("Tespit Et (Otomatik)", self)
        # self.detect_button.clicked.connect(self.detect_object)
        # self.detect_button.setEnabled(False) # Otomatik çalıştığı için devre dışı bırakılabilir
        # left_panel.addWidget(self.detect_button)

        # --- Sağ Panel (İkincil Video) ---
        right_panel = QVBoxLayout()

        # İkincil video ekranı (QLabel)
        self.secondary_video_label = QLabel("Algılanan nesne videosu burada gösterilecek", self)
        self.secondary_video_label.setAlignment(Qt.AlignCenter)
        self.secondary_video_label.setStyleSheet("background-color: #1E1E1E; color: gray; border: 1px solid gray;")
        # QSizePolicy.Ignored: Widget'ın layout içinde mümkün olduğunca genişlemesini sağlar
        self.secondary_video_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        right_panel.addWidget(self.secondary_video_label, 1) # Stretch faktörü 1 verildi

        # Açıklama etiketi (isteğe bağlı, QLabel zaten bilgi veriyor)
        # info_label = QLabel("Algılanan Nesne Videosu")
        # info_label.setAlignment(Qt.AlignCenter)
        # right_panel.addWidget(info_label)

        # Panelleri ana layouta ekle
        main_layout.addLayout(left_panel, 1) # Sol panel %50 genişlik

        # Ayırıcı Çizgi (opsiyonel ama görsel olarak ayırır)
        separator = QFrame()
        separator.setFrameShape(QFrame.VLine) # Dikey çizgi
        separator.setFrameShadow(QFrame.Sunken)
        main_layout.addWidget(separator) # Çizgiyi ekle

        main_layout.addLayout(right_panel, 1) # Sağ panel %50 genişlik

        self.setLayout(main_layout) # Ana layout'u ayarla

    def create_video_map(self):
        """VIDEO_FOLDER içindeki videoları tarayarak video_map oluşturur."""
        self.video_map = {}
        if not os.path.isdir(VIDEO_FOLDER):
            print(f"Uyarı: Video klasörü '{VIDEO_FOLDER}' bulunamadı.")
            return

        print(f"Video klasörü '{VIDEO_FOLDER}' taranıyor...")
        valid_extensions = ('.mp4', '.avi', '.mov', '.mkv') # Desteklenen video uzantıları
        for filename in os.listdir(VIDEO_FOLDER):
            if filename.lower().endswith(valid_extensions):
                # Dosya adından sınıf adını çıkar (örn: "person.mp4" -> "person")
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
            # Belki kamera bağlantısı kesildi, yeniden bağlanmayı dene?
            # self.videoCapture.release()
            # self.videoCapture = cv2.VideoCapture(CAMERA_INDEX)
            # if not self.videoCapture.isOpened(): ...
            return # Kare okunamadıysa devam etme

        # Kareyi kopyala (orijinalini bozmamak için)
        processed_frame = frame.copy()

        # YOLO modelini çalıştır (güven eşiği ile)
        results = model(processed_frame, conf=CONFIDENCE_THRESHOLD, verbose=False)[0] # verbose=False daha az konsol çıktısı verir

        # Tespit edilen nesne var mı kontrol et
        self.object_detected_in_frame = len(results.boxes) > 0

        # İşlenmiş kareyi al (kutular çizilmiş hali)
        processed_frame = self.process_detections(processed_frame, results)

        # Ana videoyu GUI'de göster
        self.display_image(processed_frame, self.video_label)

        # Eğer bu karede nesne algılanmadıysa ve ikincil video oynuyorsa durdur
        if not self.object_detected_in_frame:
            self.stop_secondary_video()
            self.features_label.clear() # Nesne yoksa özellikleri temizle
            # İkincil video alanını da temizleyebiliriz
            self.clear_label(self.secondary_video_label, "Algılanan nesne videosu burada gösterilecek")


    def process_detections(self, frame, results):
        """Algılanan nesneleri işler, kutu çizer, bilgileri gösterir ve videoyu tetikler."""
        object_info_text = ""

        if not results.boxes: # Eğer hiç nesne bulunamadıysa
             # current_playing_class'ı sıfırla ki bir sonraki tespitte video oynatılabilsin
             self.current_playing_class = None
             return frame # Çizim yapmadan orijinal (veya son işlenmiş) kareyi döndür

        # İlk tespit edilen nesne üzerinden işlem yapalım (diğerleri sadece çizilecek)
        first_box = results.boxes[0]
        cls = int(first_box.cls)
        conf = float(first_box.conf[0])
        label = model.names[cls] if cls < len(model.names) else f"Unknown ({cls})"

        # Nesne özelliklerini oluştur
        box_coords = first_box.xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = box_coords
        width = x2 - x1
        height = y2 - y1
        object_info_text = (
            f"Object Detected: {label}\n"
            f"Confidence: {conf:.2f}\n"
            f"Position (Top-Left): ({x1}, {y1})\n"
            f"Size (W x H): {width} x {height}"
        )

        # Nesne sınıfına göre ilgili videoyu oynat (eğer zaten oynamıyorsa)
        self.play_video_based_on_class(label)

        # Tüm tespit edilen nesneler için kutu ve etiket çiz
        for box in results.boxes:
            cls_i = int(box.cls)
            label_i = model.names[cls_i] if cls_i < len(model.names) else f"Unknown ({cls_i})"
            coords_i = box.xyxy[0].cpu().numpy().astype(int)
            x1_i, y1_i, x2_i, y2_i = coords_i
            conf_i = float(box.conf[0])

            # Dikdörtgen çiz
            cv2.rectangle(frame, (x1_i, y1_i), (x2_i, y2_i), (0, 255, 0), 2)
            # Etiket metnini hazırla (sınıf + güven skoru)
            text = f"{label_i}: {conf_i:.2f}"
            # Metin boyutunu hesapla
            (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            # Metin arka planını çiz
            cv2.rectangle(frame, (x1_i, y1_i - text_height - baseline), (x1_i + text_width, y1_i), (0, 255, 0), -1) # Dolgulu arka plan
            # Metni yaz
            cv2.putText(frame, text, (x1_i, y1_i - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1) # Siyah metin

        # Sadece ilk nesnenin detaylı bilgisini gösterelim
        self.features_label.setText(object_info_text)

        return frame # Kutuları çizilmiş kareyi döndür


    def play_video_based_on_class(self, object_class):
        """Algılanan nesne sınıfına göre ilgili videoyu sağ panelde oynatır."""
        object_class_lower = object_class.lower() # Küçük harfe çevirerek eşleştirmeyi kolaylaştır

        # Eğer aynı sınıfın videosu zaten oynuyorsa veya başlatılmışsa tekrar başlatma
        if self.current_playing_class == object_class_lower:
            return

        # Video haritasında bu sınıf için bir video var mı kontrol et
        video_path = self.video_map.get(object_class_lower)

        if video_path and os.path.exists(video_path):
            # Önceki videoyu durdur ve kaynakları serbest bırak
            self.stop_secondary_video(clear_label=False) # Label'ı hemen temizleme, yeni video başlayacak

            print(f"'{object_class}' algılandı. Video oynatılıyor: {video_path}")
            self.current_playing_class = object_class_lower # Şu an oynatılan sınıfı güncelle
            self.play_video(video_path)
        else:
            # Eğer bu sınıf için video yoksa veya dosya bulunamadıysa
            if self.current_playing_class is not None: # Farklı bir video oynuyorduysa durdur
                 self.stop_secondary_video()
                 self.clear_label(self.secondary_video_label, f"'{object_class}' için video bulunamadı.")
            self.current_playing_class = None # Video oynatılamadığı için sınıfı sıfırla


    def play_video(self, video_path):
        """Verilen video dosyasını ikincil video panelinde oynatmaya başlar."""
        try:
            self.secondary_video_capture = cv2.VideoCapture(video_path)
            if not self.secondary_video_capture.isOpened():
                raise IOError(f"Video dosyası açılamadı: {video_path}")

            # Video oynatma için ayrı bir timer (varsa yeniden kullan, yoksa oluştur)
            if self.secondary_timer is None:
                self.secondary_timer = QTimer(self)
                self.secondary_timer.timeout.connect(self.update_secondary_frame)

            # Timer'ı başlat (veya yeniden başlat)
            fps = self.secondary_video_capture.get(cv2.CAP_PROP_FPS)
            interval = int(1000 / fps) if fps > 0 else TIMER_INTERVAL_MS # Video FPS'ine göre ayarla
            self.secondary_timer.start(interval)

        except Exception as e:
            print(f"Hata: Video oynatma başlatılamadı: {e}")
            self.stop_secondary_video() # Hata durumunda temizle


    def update_secondary_frame(self):
        """İkincil video panelindeki kareyi günceller."""
        if self.secondary_video_capture is not None and self.secondary_video_capture.isOpened():
            ret, frame = self.secondary_video_capture.read()

            if ret:
                # Video karesini sağdaki GUI elemanında göster
                self.display_image(frame, self.secondary_video_label)
            else:
                # Video bitti veya kare okunamadı
                print(f"'{self.current_playing_class}' videosu bitti.")
                self.stop_secondary_video()
                # İsteğe bağlı: Video bitince son kare yerine mesaj gösterilebilir
                # self.clear_label(self.secondary_video_label, f"'{self.current_playing_class}' videosu tamamlandı.")
                self.current_playing_class = None # Video bittiği için sınıfı sıfırla
        else:
            # Capture nesnesi yoksa veya kapalıysa timer'ı durdur (güvenlik önlemi)
             self.stop_secondary_video()


    def stop_secondary_video(self, clear_label=True):
        """İkincil video oynatmayı durdurur ve kaynakları serbest bırakır."""
        if self.secondary_timer is not None and self.secondary_timer.isActive():
            self.secondary_timer.stop()
            # print("İkincil video timer durduruldu.") # Debug için

        if self.secondary_video_capture is not None:
            self.secondary_video_capture.release()
            self.secondary_video_capture = None
            # print("İkincil video capture serbest bırakıldı.") # Debug için

        # Eğer çağrılırken belirtilmişse veya varsayılan olarak Label'ı temizle
        if clear_label:
             self.clear_label(self.secondary_video_label, "Algılanan nesne videosu burada gösterilecek")
             self.current_playing_class = None # Label temizlendiğinde sınıfı da sıfırla


    def display_image(self, frame, label_widget):
        """Verilen OpenCV karesini belirtilen QLabel widget'ında gösterir."""
        try:
            # Kareyi QImage formatına dönüştür
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

            # QLabel boyutuna göre QPixmap'ı ölçeklendir (en/boy oranını koru)
            pixmap = QPixmap.fromImage(qt_image)
            scaled_pixmap = pixmap.scaled(
                label_widget.size(), # QLabel'in mevcut boyutunu kullan
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            label_widget.setPixmap(scaled_pixmap)
        except Exception as e:
            print(f"Hata: Görüntü gösterilirken hata oluştu: {e}")


    def clear_label(self, label_widget, text=""):
        """QLabel içeriğini temizler ve isteğe bağlı bir metin gösterir."""
        # Boş siyah bir QPixmap oluşturup ayarlayabiliriz
        # Veya sadece metni güncelleyebiliriz
        # label_widget.clear() # Sadece pixmap'i temizler
        label_widget.setText(text) # Metni ayarla (varsayılan arka plan rengiyle görünür)
        # İsterseniz tekrar siyah arka plan ayarlayabilirsiniz:
        # label_widget.setStyleSheet("background-color: #1E1E1E; color: gray; border: 1px solid gray;")


    def resizeEvent(self, event):
        """Pencere yeniden boyutlandırıldığında çağrılır."""
        # QSizePolicy.Ignored kullandığımız için QLabel'ler otomatik olarak
        # layout tarafından yeniden boyutlandırılacak. display_image içindeki
        # label_widget.size() çağrısı güncel boyutu alacağı için
        # buradaki manuel pixmap ölçeklendirmesi artık gerekli değil.
        # Sadece temel resize olayını çağırmak yeterli.
        super().resizeEvent(event)
        # print(f"Resize: Video Label: {self.video_label.size()}, Secondary: {self.secondary_video_label.size()}") # Boyutları kontrol etmek için

    # detect_object butonu otomatik çalıştığı için bu fonksiyon artık gereksiz olabilir
    # def detect_object(self):
    #     """Tespit butonuna basıldığında (artık otomatik)."""
    #     self.features_label.setText("Tespit otomatik olarak çalışıyor...")

    def closeEvent(self, event):
        """Uygulama kapatıldığında kaynakları serbest bırakır."""
        print("Uygulama kapatılıyor, kaynaklar serbest bırakılıyor...")
        self.timer.stop() # Ana timer'ı durdur
        self.stop_secondary_video() # İkincil videoyu ve timer'ı durdur

        if self.videoCapture.isOpened():
            self.videoCapture.release() # Ana kamerayı serbest bırak
            print("Ana kamera serbest bırakıldı.")

        event.accept() # Kapatma olayını kabul et


if __name__ == '__main__':
    # Video klasörünün varlığını kontrol et
    if not os.path.isdir(VIDEO_FOLDER):
        print(f"Uyarı: '{VIDEO_FOLDER}' adında bir klasör bulunamadı.")
        print("Lütfen algılanan nesneler için (örn: person.mp4, car.mp4) videoları içeren bu klasörü oluşturun.")
        # İsteğe bağlı: Klasör yoksa oluştur
        # try:
        #     os.makedirs(VIDEO_FOLDER)
        #     print(f"'{VIDEO_FOLDER}' klasörü oluşturuldu.")
        # except OSError as e:
        #     print(f"Hata: '{VIDEO_FOLDER}' klasörü oluşturulamadı: {e}")

    app = QApplication(sys.argv)
    window = ObjectDetectionApp()
    window.show()
    sys.exit(app.exec())