import os
import json
import cv2
import numpy as np
import time
import datetime
from pathlib import Path
from PySide6.QtGui import QImage, QPixmap, QColor
from PySide6.QtWidgets import QSystemTrayIcon, QApplication, QStyle

# App settings
APP_NAME = "Professional Vision Assistant"
APP_VERSION = "1.0.0"
SETTINGS_FILE = "app_settings.json"
LOG_FILE = "app_log.txt"
SCREENSHOT_FOLDER = "screenshots"
RECORDING_FOLDER = "recordings"
MODELS_FOLDER = "models"
VIDEO_FOLDER = "video"

# Create necessary folders
for folder in [SCREENSHOT_FOLDER, RECORDING_FOLDER, MODELS_FOLDER, VIDEO_FOLDER]:
    Path(folder).mkdir(exist_ok=True)

def load_settings():
    """Load application settings from JSON file"""
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            log_message(f"Error loading settings: {e}", "ERROR")
    return {
        "camera_index": 0,
        "confidence_threshold": 0.6,
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
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
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

def save_screenshot(frame, folder=SCREENSHOT_FOLDER):
    """Save a screenshot of the current frame"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(folder, f"screenshot_{timestamp}.png")
    cv2.imwrite(filename, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    return filename

def convert_cv_qt(cv_img):
    """Convert OpenCV image to QPixmap"""
    h, w, ch = cv_img.shape
    bytes_per_line = ch * w
    convert_to_qt_format = QImage(cv_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(convert_to_qt_format)

def log_message(message, level="INFO"):
    """Log a message to the log file"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] [{level}] {message}\n")

def show_notification(title, message, icon=QSystemTrayIcon.Information):
    """Show a system tray notification"""
    if QSystemTrayIcon.isSystemTrayAvailable():
        tray = QSystemTrayIcon()
        tray.setIcon(QApplication.style().standardIcon(QStyle.SP_ComputerIcon))
        tray.show()
        tray.showMessage(title, message, icon, 3000)

def get_available_models():
    """Get list of available YOLO models"""
    ensure_folders()
    models = []
    for file in os.listdir(MODELS_FOLDER):
        if file.endswith('.pt'):
            models.append(os.path.join(MODELS_FOLDER, file))
    return models

def load_json_data():
    """Load object information from JSON file"""
    try:
        with open('object_info.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        log_message(f"Error loading JSON data: {e}", "ERROR")
        return {}

def create_video_map():
    """Create mapping of class names to video files"""
    video_map = {}
    video_folder = "videos"
    
    if os.path.exists(video_folder):
        for file in os.listdir(video_folder):
            if file.endswith(('.mp4', '.avi', '.mov')):
                class_name = os.path.splitext(file)[0]
                video_map[class_name] = os.path.join(video_folder, file)
    
    return video_map 