from PySide6.QtCore import Qt, QPropertyAnimation, Property, QEasingCurve, QPoint
from PySide6.QtGui import (QPainter, QLinearGradient, QBrush, QColor, 
                          QPen, QRadialGradient, QPainterPath)
from PySide6.QtWidgets import (QFrame, QPushButton, QLabel, QVBoxLayout, 
                              QHBoxLayout, QTextEdit, QLineEdit, QComboBox,
                              QGraphicsDropShadowEffect, QSizePolicy, QWidget,
                              QScrollArea)

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

class ChatPanel(QFrame):
    """Chat panel with history and input"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("chatPanel")
        self.setStyleSheet("""
            #chatPanel {
                background-color: rgba(52, 52, 78, 180);
                border-radius: 8px;
                border: 1px solid rgba(114, 159, 207, 120);
            }
        """)
        
        layout = QVBoxLayout(self)
        
        # Chat history
        self.chat_history = QTextEdit()
        self.chat_history.setReadOnly(True)
        self.chat_history.setStyleSheet("""
            QTextEdit {
                background-color: rgba(52, 52, 78, 180);
                color: #E0E0E0;
                border: none;
                border-radius: 8px;
                padding: 10px;
            }
        """)
        
        # Chat input
        input_layout = QHBoxLayout()
        self.chat_input = QLineEdit()
        self.chat_input.setPlaceholderText("Nesne hakkında soru sorun...")
        self.chat_input.setStyleSheet("""
            QLineEdit {
                background-color: #3C6E71;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px;
            }
        """)
        
        send_button = AnimatedButton("Gönder")
        
        input_layout.addWidget(self.chat_input)
        input_layout.addWidget(send_button)
        
        layout.addWidget(self.chat_history)
        layout.addLayout(input_layout)
        
        # Connect signals
        self.chat_input.returnPressed.connect(self.on_send)
        send_button.clicked.connect(self.on_send)
        
    def on_send(self):
        """Handle send button click or enter press"""
        text = self.chat_input.text().strip()
        if text:
            self.chat_history.append(f"<b>Siz:</b> {text}")
            self.chat_input.clear()
            # Emit signal for parent to handle
            self.parent().on_chat_input(text)

class VideoPanel(QFrame):
    """Video display panel"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("videoPanel")
        self.setStyleSheet("""
            #videoPanel {
                background-color: rgba(52, 52, 78, 180);
                border-radius: 8px;
                border: 1px solid rgba(114, 159, 207, 120);
            }
        """)
        
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("Nesne Görselleştirme")
        title.setStyleSheet("""
            font-size: 16px;
            font-weight: bold;
            color: #A0C8E5;
            padding-bottom: 5px;
            border-bottom: 1px solid rgba(114, 159, 207, 120);
        """)
        title.setAlignment(Qt.AlignCenter)
        
        # Video display
        self.video_label = QLabel("Nesne tespit edildiğinde görselleştirme burada gösterilecek")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("""
            background-color: #2E2E2E;
            color: #C9D6DF;
            border-radius: 8px;
            padding: 10px;
            font-style: italic;
        """)
        self.video_label.setFixedSize(320, 240)
        self.video_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        
        # 3D model button
        self.load_model_button = AnimatedButton("3D Model Yükle")
        self.load_model_button.setStyleSheet("""
            QPushButton {
                background-color: #3C6E71;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 8px 12px;
                font-weight: bold;
                margin-top: 10px;
            }
            QPushButton:hover {
                background-color: #4D8C8F;
            }
            QPushButton:disabled {
                background-color: #2A5254;
                color: #888888;
            }
        """)
        self.load_model_button.setEnabled(False)
        
        layout.addWidget(title)
        layout.addWidget(self.video_label, 0, Qt.AlignCenter)
        layout.addWidget(self.load_model_button, 0, Qt.AlignCenter)
        
    def update_frame(self, pixmap):
        """Update video display with new frame"""
        if pixmap:
            self.video_label.setPixmap(pixmap)
            
    def set_text(self, text):
        """Set text message in video display"""
        self.video_label.setText(text)
        
    def enable_model_button(self, enabled=True):
        """Enable/disable 3D model button"""
        self.load_model_button.setEnabled(enabled)

class ControlPanel(QFrame):
    """Control panel with buttons and settings"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("controlPanel")
        self.setStyleSheet("""
            #controlPanel {
                background-color: rgba(52, 52, 78, 180);
                border-radius: 8px;
                border: 1px solid rgba(114, 159, 207, 120);
            }
        """)
        
        layout = QVBoxLayout(self)
        
        # Camera controls
        camera_controls = QHBoxLayout()
        self.screenshot_button = AnimatedButton("Ekran Görüntüsü")
        self.record_button = AnimatedButton("Kayıt Başlat")
        camera_controls.addWidget(self.screenshot_button)
        camera_controls.addWidget(self.record_button)
        
        # Model selection
        model_layout = QHBoxLayout()
        model_label = QLabel("YOLO Modeli:")
        model_label.setStyleSheet("color: #E0E0E0;")
        
        self.model_combo = QComboBox()
        self.model_combo.setStyleSheet("""
            QComboBox {
                background-color: #3C6E71;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 5px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: none;
            }
        """)
        
        self.select_model_button = AnimatedButton("Model Seç")
        
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_combo)
        model_layout.addWidget(self.select_model_button)
        
        layout.addLayout(camera_controls)
        layout.addLayout(model_layout)
        
        # Connect signals
        self.screenshot_button.clicked.connect(self.parent().take_screenshot)
        self.record_button.clicked.connect(self.parent().toggle_recording)
        self.model_combo.currentTextChanged.connect(self.parent().on_model_changed)
        self.select_model_button.clicked.connect(self.parent().select_yolo_model)
        
    def update_record_button(self, is_recording):
        """Update record button text"""
        self.record_button.setText("Kayıt Durdur" if is_recording else "Kayıt Başlat")
        
    def set_models(self, models):
        """Set available models in combo box"""
        self.model_combo.clear()
        self.model_combo.addItems(models) 