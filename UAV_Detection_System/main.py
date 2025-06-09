import sys
import os
import subprocess
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QLabel
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Nesne Tespit Sistemleri")
        self.setFixedSize(400, 300)
        
        # Ana widget ve layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Başlık
        title = QLabel("Nesne Tespit Sistemleri")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Arial", 16, QFont.Bold))
        layout.addWidget(title)
        
        # Genel nesne tespit sistemi butonu
        general_btn = QPushButton("Genel Nesne Tespit Sistemi")
        general_btn.setMinimumHeight(50)
        general_btn.clicked.connect(self.start_general_detection)
        layout.addWidget(general_btn)
        
        # UAV tespit sistemi butonu
        uav_btn = QPushButton("UAV Tespit Sistemi")
        uav_btn.setMinimumHeight(50)
        uav_btn.clicked.connect(self.start_uav_detection)
        layout.addWidget(uav_btn)
        
        # Stil
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2A2A40;
            }
            QLabel {
                color: #F0F0FA;
                margin: 20px;
            }
            QPushButton {
                background-color: #3C6E71;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 10px;
                font-size: 14px;
                margin: 10px;
            }
            QPushButton:hover {
                background-color: #4D8C8F;
            }
        """)
    
    def start_general_detection(self):
        """Genel nesne tespit sistemini başlat"""
        try:
            # Ana dizine dön
            current_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(current_dir)
            os.chdir(parent_dir)
            # Genel tespit sistemini başlat
            subprocess.Popen([sys.executable, "professional_vision_assistant.py"])
        except Exception as e:
            print(f"Error starting general detection: {e}")
    
    def start_uav_detection(self):
        """UAV tespit sistemini başlat"""
        try:
            # UAV dizinine git
            current_dir = os.path.dirname(os.path.abspath(__file__))
            os.chdir(current_dir)
            # UAV tespit sistemini başlat
            subprocess.Popen([sys.executable, "uav_detection.py"])
        except Exception as e:
            print(f"Error starting UAV detection: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec()) 