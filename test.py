from ultralytics import YOLO
import cv2


model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture(0)

while True:

    ret,frame = cap.read()
    if not ret:
        break

    results = model(frame,conf=0.5)
    boxes = results[0].boxes.xyxy  # Bounding box koordinatları (x1, y1, x2, y2)
    classes = results[0].boxes.cls  # Tespit edilen nesnelerin sınıf id'leri
    conf = results[0].boxes.conf
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        label = model.names[int(classes[i].item())]


        # Sınıf adı ve güven puanını yaz
        print(f"Nesne: {label}, Koordinatlar: ({x1}, {y1}), ({x2}, {y2})")

        # Nesnenin bulunduğu alanı çizmek
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Görüntüyü ekranda göster
        cv2.imshow("YOLO Tespit Sonuçları", frame)

# Çıkmak için 'q' tuşuna bas
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Temizleme işlemleri
cap.release()
cv2.destroyAllWindows()
