from PySide6.QtCore import Qt, QPropertyAnimation, Property, QEasingCurve, QPoint
from PySide6.QtGui import (QPainter, QLinearGradient, QBrush, QColor, 
                          QPen, QRadialGradient, QPainterPath, QGraphicsDropShadowEffect)
from PySide6.QtWidgets import QFrame, QPushButton, QLabel

# UI renk tanımlamaları
PRIMARY_COLOR = QColor(42, 42, 64)  # Dark blue
SECONDARY_COLOR = QColor(72, 72, 108)  # Medium blue
ACCENT_COLOR = QColor(114, 159, 207)  # Light blue
GRADIENT_START = QColor(35, 35, 55)  # Darker blue
GRADIENT_END = QColor(52, 52, 78)  # Lighter blue
TEXT_COLOR = QColor(240, 240, 250)  # Almost white
HIGHLIGHT_COLOR = QColor(72, 139, 220)  # Bright blue

class GradientFrame(QFrame):
    """Frame with gradient background"""
    def __init__(self, parent=None, start_color=GRADIENT_START, end_color=GRADIENT_END, 
                direction=Qt.Vertical):
        super().__init__(parent)
        self.start_color = start_color
        self.end_color = end_color
        self.direction = direction
        
    def paintEvent(self, event):
        painter = QPainter(self)
        gradient = QLinearGradient()
        
        if self.direction == Qt.Vertical:
            gradient.setStart(0, 0)
            gradient.setFinalStop(0, self.height())
        else:
            gradient.setStart(0, 0)
            gradient.setFinalStop(self.width(), 0)
        
        gradient.setColorAt(0, self.start_color)
        gradient.setColorAt(1, self.end_color)
        
        painter.fillRect(self.rect(), QBrush(gradient))

class RoundedFrame(QFrame):
    """Frame with rounded corners"""
    def __init__(self, parent=None, radius=10, bg_color=PRIMARY_COLOR):
        super().__init__(parent)
        self.radius = radius
        self.bg_color = bg_color
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Set background color
        painter.setBrush(QBrush(self.bg_color))
        
        # Set border
        painter.setPen(Qt.NoPen)
        
        # Draw rounded rectangle
        painter.drawRoundedRect(self.rect(), self.radius, self.radius)

class AnimatedButton(QPushButton):
    """Button with hover and click animations"""
    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setStyleSheet("""
            QPushButton {
                background-color: #3C6E71;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 8px 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #4D8C8F;
            }
            QPushButton:pressed {
                background-color: #2A5254;
            }
        """)
        
        # Add subtle shadow effect
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setOffset(0, 2)
        shadow.setBlurRadius(5)
        shadow.setColor(QColor(0, 0, 0, 80))
        self.setGraphicsEffect(shadow)

class InfoPanel(QFrame):
    """Information panel with title and content"""
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.setObjectName("infoPanel")
        self.setStyleSheet("""
            #infoPanel {
                background-color: rgba(52, 52, 78, 180);
                border-radius: 8px;
                border: 1px solid rgba(114, 159, 207, 120);
            }
            QLabel {
                color: #E0E0E0;
            }
        """)
        
        layout = QVBoxLayout(self)
        
        # Title
        title_label = QLabel(title)
        title_label.setStyleSheet("""
            font-size: 16px;
            font-weight: bold;
            color: #A0C8E5;
            padding-bottom: 5px;
            border-bottom: 1px solid rgba(114, 159, 207, 120);
        """)
        
        # Content
        self.content = QLabel("Bekliyor...")
        self.content.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.content.setWordWrap(True)
        
        layout.addWidget(title_label)
        layout.addWidget(self.content)
        
    def update_content(self, text):
        self.content.setText(text) 