import sys
import cv2
import numpy as np
from PySide6.QtCore import QTimer, Qt, QSize
from PySide6.QtGui import QImage, QPixmap, QFont
from PySide6.QtWidgets import (QApplication, QWidget, QLabel, QVBoxLayout,
                             QHBoxLayout, QFrame, QSizePolicy, QGroupBox)
import multiprocessing as mp
import time
from utils import (detection_process, secondary_video_process, load_model,
                  load_json_data, create_video_map, convert_cv_qt)
from Sound_Project.Sound import ses, diger

class ObjectDetectionApp(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Object Detection GUI")
        self.setGeometry(100, 100, 1200, 700)
        self.setStyleSheet("background-color: #1C2526;")  # Dark theme background

        # Initialize multiprocessing components
        self.frame_queue = mp.Queue(maxsize=2)
        self.result_queue = mp.Queue(maxsize=2)
        self.secondary_frame_queue = mp.Queue(maxsize=2)
        self.parent_conn, self.child_conn = mp.Pipe()

        # Start detection process
        self.detection_proc = mp.Process(
            target=detection_process,
            args=(self.frame_queue, self.result_queue, "yolov8n.pt", 0, 0.6)
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
        self.json_data = load_json_data("object_info.json")
        self.video_map = create_video_map("video")
        self.init_ui()

        # Sound control
        self.last_sound_time = 0
        self.last_sound_class = None

        # Timer for updating GUI
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # 30ms interval

        self.current_playing_class = None

    def init_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # --- Left Panel ---
        left_panel = QVBoxLayout()
        left_panel.setSpacing(8)

        # Title
        left_title = QLabel("Camera Feed")
        left_title.setStyleSheet("color: #C9D6DF; font-size: 16px; font-weight: bold; padding: 5px;")
        left_title.setAlignment(Qt.AlignCenter)
        left_panel.addWidget(left_title)

        # Main video screen
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

        # Object features frame
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
        self.features_label = QLabel("Waiting for object information...")
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

        # --- Right Panel ---
        right_panel = QVBoxLayout()
        right_panel.setSpacing(8)

        # Title
        right_title = QLabel("Detected Object Information")
        right_title.setStyleSheet("color: #C9D6DF; font-size: 16px; font-weight: bold; padding: 5px;")
        right_title.setAlignment(Qt.AlignCenter)
        right_panel.addWidget(right_title)

        # Secondary video screen
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

        # JSON information frame
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

        # Icon area
        self.json_icon_label = QLabel()
        self.json_icon_label.setFixedSize(48, 48)
        self.json_icon_label.setStyleSheet("background-color: transparent;")
        self.json_icon_label.setAlignment(Qt.AlignCenter)
        json_layout.addWidget(self.json_icon_label)

        # JSON information
        self.json_info_label = QLabel("Detected object information will be shown here.")
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
        separator.setStyleSheet("background-color: #52616B;")
        main_layout.addWidget(separator)
        main_layout.addLayout(right_panel, 1)

    def update_frame(self):
        try:
            # Update main video feed
            if not self.frame_queue.empty():
                frame, detections = self.frame_queue.get()
                if frame is not None:
                    self.display_image(frame, self.video_label)
                    self.process_detections(detections)

            # Update secondary video feed
            if not self.secondary_frame_queue.empty():
                frame = self.secondary_frame_queue.get()
                if frame is not None:
                    self.display_image(frame, self.secondary_video_label)

        except Exception as e:
            print(f"Error updating frame: {e}")

    def process_detections(self, detections):
        if not detections:
            return

        # Get the detection with highest confidence
        best_detection = max(detections, key=lambda x: x['conf'])
        label = best_detection['label']
        conf = best_detection['conf']

        # Update features label
        features_text = f"Detected: {label}\nConfidence: {conf:.2f}"
        self.features_label.setText(features_text)

        # Update JSON information
        if label in self.json_data:
            info = self.json_data[label]
            self.json_info_label.setText(f"Name: {info.get('name', label)}\n"
                                       f"Description: {info.get('description', 'No description available')}")
        else:
            self.json_info_label.setText(f"No additional information available for {label}")

        # Play video if available
        if label in self.video_map:
            self.play_video_based_on_class(label)

        # Play sound with cooldown
        current_time = time.time()
        if (current_time - self.last_sound_time > 2.0 or 
            self.last_sound_class != label):
            # Use the ses function for text-to-speech
            ses(f"Detected {label}")
            self.last_sound_time = current_time
            self.last_sound_class = label

    def play_video_based_on_class(self, object_class):
        if object_class != self.current_playing_class:
            if object_class in self.video_map:
                self.parent_conn.send(self.video_map[object_class])
                self.current_playing_class = object_class
            else:
                self.stop_secondary_video()

    def stop_secondary_video(self):
        self.parent_conn.send("stop")
        self.current_playing_class = None
        self.clear_label(self.secondary_video_label)

    def display_image(self, frame, label_widget):
        if frame is not None:
            # Convert frame to RGB
            rgb_frame = convert_cv_qt(frame)
            
            # Convert to QImage
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            # Scale to fit label while maintaining aspect ratio
            scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
                label_widget.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            
            label_widget.setPixmap(scaled_pixmap)

    def clear_label(self, label_widget, text=""):
        label_widget.clear()
        if text:
            label_widget.setText(text)

    def closeEvent(self, event):
        # Clean up processes
        self.detection_proc.terminate()
        self.secondary_video_proc.terminate()
        event.accept()

if __name__ == "__main__":
    # Set multiprocessing start method
    mp.set_start_method('spawn')
    
    app = QApplication(sys.argv)
    window = ObjectDetectionApp()
    window.show()
    sys.exit(app.exec()) 