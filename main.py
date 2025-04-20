from ultralytics import YOLO
import cv2
import numpy as np
import trimesh
import os
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget


# 3D model yükleme fonksiyonu
def render_obj_as_image(obj_filename):
    # OBJ dosyasını yükle
    try:
        mesh = trimesh.load(obj_filename)
        # Mesh'i 2D görüntüye dönüştür
        model_img = mesh.show()
        return model_img
    except Exception as e:
        print(f"3D model yükleme hatası: {e}")
        return None


# Pyside6 GUI sınıfı
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # GUI bileşenlerini oluştur
        self.setWindowTitle('YOLO ve 3D Model Görselleştirici')
        self.setGeometry(100, 100, 1200, 800)

        self.camera_label = QLabel(self)
        self.camera_label.setAlignment(Qt.AlignCenter)

        self.model_label = QLabel(self)
        self.model_label.setAlignment(Qt.AlignCenter)

        self.info_label = QLabel(self)
        self.info_label.setAlignment(Qt.AlignCenter)

        layout = QVBoxLayout()
        layout.addWidget(self.camera_label)
        layout.addWidget(self.model_label)
        layout.addWidget(self.info_label)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Timer ayarla
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # 30ms'lik bir gecikme (yaklaşık 30 FPS)

        # YOLO modelini yükle
        self.model = YOLO('yolov8n.pt')

        # Video kaynağını başlat
        self.cap = cv2.VideoCapture(0)

        # Önceki nesneyi takip et
        self.previous_label = None

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        # YOLO ile tahmin yap
        results = self.model(frame, conf=0.6)
        boxes = results[0].boxes.xyxy
        classes = results[0].boxes.cls  # Tespit edilen nesnelerin sınıf id'leri
        conf = results[0].boxes.conf

        # Nesne tespiti ve 3D model kontrolü
        new_model_shown = False  # Yeni model gösterildi mi?

        for i, box in enumerate(boxes):
            label = self.model.names[int(classes[i].item())]  # Sınıf ismini almak için düzeltme
            obj_filename = f"obj_file/{label}.obj"

            if os.path.exists(obj_filename):
                # 3D modeli görselleştir
                model_img = render_obj_as_image(obj_filename)
                if model_img is not None:
                    model_img = np.ascontiguousarray(model_img)
                    q_img_model = QImage(model_img, model_img.shape[1], model_img.shape[0], QImage.Format_RGB888)

                    # Eğer bu yeni bir nesne ise, önceki nesneyi temizle
                    if self.previous_label != label:
                        self.previous_label = label
                        self.model_label.setPixmap(QPixmap.fromImage(q_img_model))
                        self.info_label.setText(f"{label} bulundu.")
                        new_model_shown = True
                    break  # Sadece bir nesne göstermek için döngüden çık

        if not new_model_shown:
            # Eğer yeni bir model gösterilmediyse, eski modeli temizle
            self.model_label.clear()
            self.info_label.setText("Yeni nesne bulunamadı.")

        # Kameradan alınan görüntüyü GUI'ye aktarma
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, c = frame_rgb.shape
        q_img_camera = QImage(frame_rgb.data, w, h, w * c, QImage.Format_RGB888)
        self.camera_label.setPixmap(QPixmap.fromImage(q_img_camera))

    def closeEvent(self, event):
        self.cap.release()
        event.accept()


if __name__ == '__main__':
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()
