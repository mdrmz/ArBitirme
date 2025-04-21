import cv2
from ultralytics import YOLO
import numpy as np

# Yüz tespiti için YOLOv8 modelini yükle
model = YOLO("yolov11n-face.pt")  # Yüz tespiti için özel model

# Gözlük filtresini yükle (şeffaf PNG)
glasses_img = cv2.imread("filter/gozluk-removebg-preview.png", cv2.IMREAD_UNCHANGED)  # RGBA olarak yükle

# Kamera başlat
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]  # Yüz tespiti modelini çalıştır

    for box in results.boxes:
        cls = int(box.cls[0])  # tespit edilen sınıf
        if cls == 0:  # Sınıf 0 = yüz
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Yüzün sınırlarını al
            width = x2 - x1
            height = y2 - y1

            # Yüzün etrafına dikdörtgen çiz
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Yeşil renkte dikdörtgen

            # Gözlük boyutunu ve pozisyonunu ayarla
            gw = width  # Gözlük genişliğini yüz genişliğiyle eşitle
            gh = int(gw * glasses_img.shape[0] / glasses_img.shape[1])  # Orantılı olarak yükseklik

            # Gözlüğü yerleştireceğimiz yeni pozisyon
            x_offset = x1
            y_offset = y1 - gh / 6  # Yüzün üst kısmına yerleştir, tam sayı yerine ondalıklı bölme kullanıldı

            # Frame dışına taşmayı önle
            """if x_offset + gw > frame.shape[1] or y_offset + gh > frame.shape[0]:
                continue"""

            resized_glasses = cv2.resize(glasses_img, (gw, gh), interpolation=cv2.INTER_AREA)

            # Şeffaflık kontrolü (RGBA)
            if resized_glasses.shape[2] == 4:
                alpha_s = resized_glasses[:, :, 3] / 255.0
                alpha_l = 1.0 - alpha_s

                for c in range(3):  # BGR renk kanalları
                    frame[int(y_offset):int(y_offset + gh), x_offset:int(x_offset + gw), c] = (
                        alpha_s * resized_glasses[:, :, c] +
                        alpha_l * frame[int(y_offset):int(y_offset + gh), x_offset:int(x_offset + gw), c]
                    )
            else:
                frame[int(y_offset):int(y_offset + gh), x_offset:int(x_offset + gw)] = resized_glasses

    cv2.imshow("Filter Cam", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
