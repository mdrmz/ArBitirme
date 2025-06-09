import time
from PySide6.QtCore import QThread
from Sound_Project.Sound import ses

class SoundThread(QThread):
    """Thread for handling TTS announcements"""
    def __init__(self):
        super().__init__()
        self.last_sound_time = 0
        self.last_sound_class = None
        self.running = True
        self.queue = []
        self.sound_cooldown = 2  # seconds
        
    def add_to_queue(self, class_name):
        """Add sound notification to queue"""
        current_time = time.time()
        
        # Check cooldown and duplicate
        if (current_time - self.last_sound_time < self.sound_cooldown or 
            class_name == self.last_sound_class):
            return
            
        self.queue.append(class_name)
        self.last_sound_time = current_time
        self.last_sound_class = class_name
        
    def run(self):
        while self.running:
            if self.queue:
                try:
                    class_name = self.queue.pop(0)
                    ses(class_name)
                except Exception as e:
                    print(f"Sound error: {e}")
            time.sleep(0.1)
            
    def stop(self):
        """Stop sound thread"""
        self.running = False
        self.wait() 