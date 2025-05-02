import sys
import cv2
import numpy as np
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QApplication, QWidget, QLabel, QVBoxLayout,
                               QHBoxLayout, QFrame, QSizePolicy, QGroupBox)
from ultralytics import YOLO
import os
import json
import multiprocessing as mp
from Sound_Project import Sound
import time

# --- Configuration ---
MODEL_PATH = "yolov8n.pt"
CAMERA_INDEX = 0
TIMER_INTERVAL_MS = 30
VIDEO_FOLDER = "video"
JSON_PATH = "object_info.json"
CONFIDENCE_THRESHOLD = 0.6
SOUND_COOLDOWN = 2.0  # Seconds before repeating the same sound
# ---------------------

def detection_process(frame_queue, result_queue, model_path, camera_index, conf_threshold):
    """Process for running YOLO object detection on camera frames."""
    try:
        model = YOLO(model_path)
        video_capture = cv2.VideoCapture(camera_index)
        if not video_capture.isOpened():
            print(f"ERROR: Camera {camera_index} could not be opened in detection process.")
            return

        while True:
            ret, frame = video_capture.read()
            if not ret:
                print("ERROR: Failed to read camera frame in detection process.")
                continue

            # Run YOLO inference
            results = model(frame, conf=conf_threshold, verbose=False)[0]
            processed_frame = frame.copy()

            # Process detections
            detections = []
            for box in results.boxes:
                cls = int(box.cls)
                conf = float(box.conf[0])
                label = model.names[cls] if cls < len(model.names) else f"Unknown ({cls})"
                coords = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = coords
                width = x2 - x1
                height = y2 - y1

                # Draw bounding box and label
                cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                text = f"{label}: {conf:.2f}"
                (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(processed_frame, (x1, y1 - text_height - baseline), (x1 + text_width, y1), (0, 255, 0), -1)
                cv2.putText(processed_frame, text, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                detections.append({
                    'label': label,
                    'conf': conf,
                    'coords': (x1, y1, x2, y2),
                    'width': width,
                    'height': height
                })

            # Send frame and detections to GUI
            frame_queue.put((processed_frame, detections))
            result_queue.put(detections)

    except Exception as e:
        print(f"ERROR in detection process: {e}")
    finally:
        if 'video_capture' in locals():
            video_capture.release()

def secondary_video_process(frame_queue, conn):
    """Process for playing secondary video based on detected object class."""
    video_capture = None
    current_video_path = None
    try:
        while True:
            # Check for new video path
            if conn.poll():
                msg = conn.recv()
                if msg == "stop":
                    if video_capture is not None:
                        video_capture.release()
                        video_capture = None
                        current_video_path = None
                    continue
                elif isinstance(msg, str):
                    if msg != current_video_path:
                        if video_capture is not None:
                            video_capture.release()
                        video_capture = cv2.VideoCapture(msg)
                        if not video_capture.isOpened():
                            print(f"ERROR: Could not open video {msg}")
                            video_capture = None
                            continue
                        current_video_path = msg

            if video_capture is not None and video_capture.isOpened():
                ret, frame = video_capture.read()
                if ret:
                    frame_queue.put(frame)
                else:
                    # Loop video
                    video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            else:
                # No video playing, sleep briefly to avoid busy loop
                time.sleep(0.01)

    except Exception as e:
        print(f"ERROR in secondary video process: {e}")
    finally:
        if video_capture is not None:
            video_capture.release()

class ObjectDetectionApp(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Object Detection GUI")
        self.setGeometry(100, 100, 1200, 700)
        self.setStyleSheet("background-color: #1C2526;")

        # Initialize multiprocessing components
        self.frame_queue = mp.Queue(maxsize=2)
        self.result_queue = mp.Queue(maxsize=2)
        self.secondary_frame_queue = mp.Queue(maxsize=2)
        self.parent_conn, self.child_conn = mp.Pipe()

        # Start detection process
        self.detection_proc = mp.Process(
            target=detection_process,
            args=(self.frame_queue, self.result_queue, MODEL_PATH, CAMERA_INDEX, CONFIDENCE_THRESHOLD)
        )
        self.detection_proc.daemon = True
        self.detection_proc.start()

        # Start secondary video process
        self.secondary_video_proc = mp.Process(
            target=secondary_video_process,
            args=(self.secondary_frame_queue, self.child_conn)
        )
        self.secondary_video_proc.daemon = True
        self.secondary_video_proc.start()

        # Initialize UI
        self.json_data = self.load_json_data()
        self.init_ui()
        self.create_video_map()

        # Sound control
        self.last_sound_time = 0
        self.last_sound_class = None

        # Timer for updating GUI
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(TIMER_INTERVAL_MS)

        self.current_playing_class = None

    def load_json_data(self):
        """Load JSON data from file."""
        try:
            if not os.path.exists(JSON_PATH):
                print(f"WARNING: JSON file '{JSON_PATH}' not found.")
                return {}
            with open(JSON_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if "objects" not in data:
                    raise ValueError("JSON file does not contain 'objects' key.")
                print("JSON data loaded successfully:", list(data["objects"].keys()))
                return data["objects"]
        except Exception as e:
            print(f"ERROR: Failed to load JSON file: {e}")
            return {}

    def init_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # Left Panel
        left_panel = QVBoxLayout()
        left_panel.setSpacing(8)

        left_title = QLabel("Camera Feed")
        left_title.setStyleSheet("color: #C9D6DF; font-size: 16px; font-weight: bold; padding: 5px;")
        left_title.setAlignment(Qt.AlignCenter)
        left_panel.addWidget(left_title)

        self.video_label = QLabel("Waiting for camera...", self)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("""
            background-color: #2E2E2E;
            color: #C9D6DF;
            border: 2px solid #52616B;
            border-radius: 8px;
        """)
        self.video_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        left_panel.addWidget(self.video_label, 1)

        features_frame = QGroupBox("Object Features")
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
        self.features_label = QLabel("Waiting for object info...")
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

        # Right Panel
        right_panel = QVBoxLayout()
        right_panel.setSpacing(8)

        right_title = QLabel("Detected Object Info")
        right_title.setStyleSheet("color: #C9D6DF; font-size: 16px; font-weight: bold; padding: 5px;")
        right_title.setAlignment(Qt.AlignCenter)
        right_panel.addWidget(right_title)

        self.secondary_video_label = QLabel("Detected object video will be shown here", self)
        self.secondary_video_label.setAlignment(Qt.AlignCenter)
        self.secondary_video_label.setStyleSheet("""
            background-color: #2E2E2E;
            color: #C9D6DF;
            border: 2px solid #52616B;
            border-radius: 8px;
        """)
        self.secondary_video_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        right_panel.addWidget(self.secondary_video_label, 1)

        json_frame = QGroupBox("Object Details")
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

        self.json_icon_label = QLabel()
        self.json_icon_label.setFixedSize(48, 48)
        self.json_icon_label.setStyleSheet("background-color: transparent;")
        self.json_icon_label.setAlignment(Qt.AlignCenter)
        json_layout.addWidget(self.json_icon_label)

        self.json_info_label = QLabel("Detected object info will be shown here.")
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

        # Add panels to main layout
        main_layout.addLayout(left_panel, 1)
        separator = QFrame()
        separator.setFrameShape(QFrame.VLine)
        separator.setFrameShadow(QFrame.Sunken)
        separator.setStyleSheet("color: #52616B;")
        main_layout.addWidget(separator)
        main_layout.addLayout(right_panel, 1)
        self.setLayout(main_layout)

    def create_video_map(self):
        """Scan VIDEO_FOLDER and create video_map."""
        self.video_map = {}
        if not os.path.isdir(VIDEO_FOLDER):
            print(f"WARNING: Video folder '{VIDEO_FOLDER}' not found.")
            return

        print(f"Scanning video folder '{VIDEO_FOLDER}'...")
        valid_extensions = ('.mp4', '.avi', '.mov', '.mkv')
        for filename in os.listdir(VIDEO_FOLDER):
            if filename.lower().endswith(valid_extensions):
                class_name = os.path.splitext(filename)[0].lower()
                video_path = os.path.join(VIDEO_FOLDER, filename)
                self.video_map[class_name] = video_path
                print(f"  -> Found video: '{class_name}' -> '{video_path}'")
        print("Video map created:", self.video_map)

    def update_frame(self):
        """Update the main camera feed and process detections."""
        # Update main video feed
        if not self.frame_queue.empty():
            frame, detections = self.frame_queue.get()
            self.display_image(frame, self.video_label)

        # Update secondary video feed
        if not self.secondary_frame_queue.empty():
            frame = self.secondary_frame_queue.get()
            self.display_image(frame, self.secondary_video_label)

        # Process detections
        if not self.result_queue.empty():
            detections = self.result_queue.get()
            if not detections:
                self.stop_secondary_video()
                self.features_label.setText("Waiting for object info...")
                self.json_info_label.setText("Detected object info will be shown here.")
                self.json_icon_label.clear()
                self.current_playing_class = None
                return

            # Process first detection
            detection = detections[0]
            label = detection['label']
            conf = detection['conf']
            x1, y1, x2, y2 = detection['coords']
            width = detection['width']
            height = detection['height']

            # Handle sound
            current_time = time.time()
            if (label != self.last_sound_class or
                current_time - self.last_sound_time > SOUND_COOLDOWN):
                Sound.ses(label)
                self.last_sound_time = current_time
                self.last_sound_class = label

            # Update feature labels
            object_info_text = (
                f"<b>Object:</b> {label}<br>"
                f"<b>Confidence:</b> {conf:.2f}<br>"
                f"<b>Position (Top-Left):</b> ({x1}, {y1})<br>"
                f"<b>Size (W x H):</b> {width} x {height}"
            )
            self.features_label.setText(object_info_text)

            # Update JSON info
            label_lower = label.lower()
            if label_lower in self.json_data:
                json_info = self.json_data[label_lower]
                json_info_text = (
                    f"<b>Name:</b> {json_info.get('display_name', label)}<br>"
                    f"<b>Description:</b> {json_info.get('description', 'No info')}<br>"
                    f"<b>Extra Info:</b> {json_info.get('extra_info', 'No info')}"
                )
                icon_path = json_info.get('icon_path')
                if icon_path and os.path.exists(icon_path):
                    pixmap = QPixmap(icon_path).scaled(48, 48, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    self.json_icon_label.setPixmap(pixmap)
                else:
                    self.json_icon_label.clear()
            else:
                json_info_text = f"No JSON data found for '{label}'."
                self.json_icon_label.clear()
            self.json_info_label.setText(json_info_text)

            # Play video based on class
            self.play_video_based_on_class(label)

    def play_video_based_on_class(self, object_class):
        """Play video corresponding to detected object class."""
        object_class_lower = object_class.lower()
        if self.current_playing_class == object_class_lower:
            return

        video_path = self.video_map.get(object_class_lower)
        if video_path and os.path.exists(video_path):
            print(f"'{object_class}' detected. Playing video: {video_path}")
            self.parent_conn.send(video_path)
            self.current_playing_class = object_class_lower
        else:
            self.stop_secondary_video()
            self.clear_label(self.secondary_video_label, f"No video found for '{object_class}'.")
            self.current_playing_class = None

    def stop_secondary_video(self):
        """Stop secondary video playback."""
        self.parent_conn.send("stop")
        self.clear_label(self.secondary_video_label, "Detected object video will be shown here")

    def display_image(self, frame, label_widget):
        """Display OpenCV frame in the specified QLabel widget."""
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
            print(f"ERROR: Failed to display image: {e}")

    def clear_label(self, label_widget, text=""):
        """Clear QLabel content and set optional text."""
        label_widget.setText(text)

    def closeEvent(self, event):
        """Clean up resources when closing the application."""
        print("Closing application, releasing resources...")
        self.timer.stop()
        self.stop_secondary_video()

        # Terminate processes
        self.detection_proc.terminate()
        self.secondary_video_proc.terminate()
        self.detection_proc.join()
        self.secondary_video_proc.join()

        # Close queues
        self.frame_queue.close()
        self.result_queue.close()
        self.secondary_frame_queue.close()

        event.accept()

if __name__ == '__main__':
    if not os.path.isdir(VIDEO_FOLDER):
        print(f"WARNING: Folder '{VIDEO_FOLDER}' not found.")
        print("Please create this folder with videos for detected objects (e.g., person.mp4, car.mp4).")

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = ObjectDetectionApp()
    window.show()
    sys.exit(app.exec())