import cv2
import numpy as np
import json
import os
import multiprocessing as mp
import time
from typing import Dict, List, Tuple, Optional, Any
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QApplication, QWidget, QLabel, QVBoxLayout,
                             QHBoxLayout, QFrame, QSizePolicy, QGroupBox)

# --- Configuration ---
MODEL_PATH = "yolov8n.pt"
CAMERA_INDEX = 0
TIMER_INTERVAL_MS = 30
VIDEO_FOLDER = "video"
JSON_PATH = "object_info.json"
CONFIDENCE_THRESHOLD = 0.6
SOUND_COOLDOWN = 2.0  # Seconds before repeating the same sound
# ---------------------

def detection_process(frame_queue: mp.Queue, result_queue: mp.Queue, model_path: str, 
                     camera_index: int, conf_threshold: float) -> None:
    """
    Process for running YOLO object detection on camera frames.
    
    Args:
        frame_queue: Queue for sending processed frames
        result_queue: Queue for sending detection results
        model_path: Path to YOLO model
        camera_index: Camera device index
        conf_threshold: Confidence threshold for detections
    """
    try:
        from ultralytics import YOLO
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

                # Draw bounding box and label with enhanced aesthetics
                cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                text = f"{label}: {conf:.2f}"
                (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(processed_frame, (x1, y1 - text_height - baseline), 
                            (x1 + text_width, y1), (0, 255, 0), -1)
                cv2.putText(processed_frame, text, (x1, y1 - baseline), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

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

def secondary_video_process(frame_queue: mp.Queue, conn: mp.Pipe) -> None:
    """
    Process for playing secondary video based on detected object class.
    
    Args:
        frame_queue: Queue for sending video frames
        conn: Pipe for receiving video path commands
    """
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

def load_model(model_path: str) -> Optional[object]:
    """
    Load a YOLO model from the given path.
    
    Args:
        model_path (str): Path to the YOLO model file
        
    Returns:
        object: Loaded YOLO model or None if loading fails
    """
    try:
        from ultralytics import YOLO
        model = YOLO(model_path)
        if not hasattr(model, 'names') or not model.names:
            raise ValueError("Model names not loaded or empty")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def load_json_data(json_path: str) -> Dict:
    """
    Load and validate JSON data from a file.
    
    Args:
        json_path (str): Path to the JSON file
        
    Returns:
        Dict: Loaded JSON data or empty dict if loading fails
    """
    try:
        if not os.path.exists(json_path):
            print(f"Warning: JSON file '{json_path}' not found")
            return {}
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if "objects" not in data:
                raise ValueError("'objects' key not found in JSON file")
            return data["objects"]
    except Exception as e:
        print(f"Error loading JSON data: {e}")
        return {}

def process_frame(frame: np.ndarray, model: object, confidence_threshold: float = 0.6) -> Tuple[np.ndarray, List]:
    """
    Process a frame using the YOLO model and draw detections with enhanced aesthetics.
    
    Args:
        frame (np.ndarray): Input frame
        model (object): YOLO model
        confidence_threshold (float): Minimum confidence threshold for detections
        
    Returns:
        Tuple[np.ndarray, List]: Processed frame and list of detections
    """
    try:
        results = model(frame, conf=confidence_threshold)[0]
        detections = []
        
        for box in results.boxes:
            cls = int(box.cls)
            conf = float(box.conf[0])
            label = model.names[cls] if cls < len(model.names) else f"Unknown ({cls})"
            coords = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = coords
            width = x2 - x1
            height = y2 - y1

            # Enhanced bounding box and label drawing
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f"{label}: {conf:.2f}"
            (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (x1, y1 - text_height - baseline), 
                        (x1 + text_width, y1), (0, 255, 0), -1)
            cv2.putText(frame, text, (x1, y1 - baseline), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            detections.append({
                'label': label,
                'conf': conf,
                'coords': (x1, y1, x2, y2),
                'width': width,
                'height': height
            })
        
        return frame, detections
    except Exception as e:
        print(f"Error processing frame: {e}")
        return frame, []

def get_video_info(video_path: str) -> Tuple[int, int, float]:
    """
    Get video information including width, height, and FPS.
    
    Args:
        video_path (str): Path to the video file
        
    Returns:
        Tuple[int, int, float]: Width, height, and FPS of the video
    """
    try:
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return width, height, fps
    except Exception as e:
        print(f"Error getting video info: {e}")
        return 0, 0, 0.0

def convert_cv_qt(cv_img: np.ndarray, target_size: Tuple[int, int] = None) -> np.ndarray:
    """
    Convert OpenCV image to Qt-compatible format with optional resizing.
    
    Args:
        cv_img (np.ndarray): OpenCV image
        target_size (Tuple[int, int], optional): Target size for resizing
        
    Returns:
        np.ndarray: Converted image
    """
    try:
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        
        if target_size:
            rgb_image = cv2.resize(rgb_image, target_size)
            
        return rgb_image
    except Exception as e:
        print(f"Error converting image: {e}")
        return cv_img

def create_video_map(video_folder: str) -> Dict[str, str]:
    """
    Create a mapping of object classes to their corresponding video files.
    
    Args:
        video_folder (str): Path to the folder containing video files
        
    Returns:
        Dict[str, str]: Mapping of object classes to video file paths
    """
    video_map = {}
    try:
        if not os.path.exists(video_folder):
            print(f"Warning: Video folder '{video_folder}' not found")
            return video_map

        for filename in os.listdir(video_folder):
            if filename.endswith(('.mp4', '.avi', '.mov')):
                class_name = os.path.splitext(filename)[0]
                video_map[class_name] = os.path.join(video_folder, filename)
        
        return video_map
    except Exception as e:
        print(f"Error creating video map: {e}")
        return video_map 