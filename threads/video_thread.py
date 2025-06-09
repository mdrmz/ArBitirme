import cv2
import numpy as np
import time
import os
from PySide6.QtCore import QThread, Signal

class VideoThread(QThread):
    """Thread for playing class-specific videos"""
    frame_ready = Signal(np.ndarray)
    error_occurred = Signal(str)
    
    def __init__(self):
        super().__init__()
        self.video_path = None
        self.running = True
        self.cap = None
        
    def set_video(self, video_path):
        """Set video file to play"""
        if self.cap:
            self.cap.release()
        self.video_path = video_path
        
    def run(self):
        while self.running:
            if not self.video_path or not os.path.exists(self.video_path):
                time.sleep(0.1)
                continue
                
            try:
                if not self.cap:
                    self.cap = cv2.VideoCapture(self.video_path)
                    if not self.cap.isOpened():
                        raise Exception("Video dosyası açılamadı")
                        
                ret, frame = self.cap.read()
                if not ret:
                    # Video bitti, başa dön
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                    
                # Convert BGR to RGB for Qt
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.frame_ready.emit(frame_rgb)
                
                # Control playback speed
                time.sleep(1/30)  # 30 FPS
                
            except Exception as e:
                self.error_occurred.emit(str(e))
                time.sleep(1)  # Wait before retrying
                
    def stop(self):
        """Stop video playback"""
        self.running = False
        if self.cap:
            self.cap.release()
        self.wait() 