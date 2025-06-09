from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                              QComboBox, QDoubleSpinBox, QCheckBox, QGroupBox,
                              QGridLayout)
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor

from gui_components import AnimatedButton
from utils.helpers import load_settings, save_settings, get_available_cameras

class SettingsDialog(QDialog):
    """Settings dialog for the application"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Ayarlar")
        self.setModal(True)
        self.resize(400, 500)
        
        # Load current settings
        self.settings = load_settings()
        
        # Create UI
        self.init_ui()
    
    def init_ui(self):
        """Initialize the settings UI"""
        layout = QVBoxLayout(self)
        
        # Camera settings
        camera_group = QGroupBox("Kamera Ayarları")
        camera_layout = QGridLayout()
        
        camera_label = QLabel("Kamera Seçimi:")
        self.camera_combo = QComboBox()
        self.camera_combo.addItems([f"Kamera {i}" for i in get_available_cameras()])
        self.camera_combo.setCurrentIndex(self.settings["camera_index"])
        
        camera_layout.addWidget(camera_label, 0, 0)
        camera_layout.addWidget(self.camera_combo, 0, 1)
        
        camera_group.setLayout(camera_layout)
        layout.addWidget(camera_group)
        
        # Detection settings
        detection_group = QGroupBox("Algılama Ayarları")
        detection_layout = QGridLayout()
        
        conf_label = QLabel("Güven Eşiği:")
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.1, 1.0)
        self.conf_spin.setSingleStep(0.1)
        self.conf_spin.setValue(self.settings["confidence_threshold"])
        
        detection_layout.addWidget(conf_label, 0, 0)
        detection_layout.addWidget(self.conf_spin, 0, 1)
        
        detection_group.setLayout(detection_layout)
        layout.addWidget(detection_group)
        
        # Feature settings
        feature_group = QGroupBox("Özellik Ayarları")
        feature_layout = QGridLayout()
        
        self.sound_check = QCheckBox("Ses Bildirimleri")
        self.sound_check.setChecked(self.settings["sound_enabled"])
        
        self.video_check = QCheckBox("Video Oynatma")
        self.video_check.setChecked(self.settings["video_enabled"])
        
        self.auto_screenshot_check = QCheckBox("Otomatik Ekran Görüntüsü")
        self.auto_screenshot_check.setChecked(self.settings["auto_save_screenshots"])
        
        self.recording_check = QCheckBox("Otomatik Kayıt")
        self.recording_check.setChecked(self.settings["recording_enabled"])
        
        feature_layout.addWidget(self.sound_check, 0, 0)
        feature_layout.addWidget(self.video_check, 0, 1)
        feature_layout.addWidget(self.auto_screenshot_check, 1, 0)
        feature_layout.addWidget(self.recording_check, 1, 1)
        
        feature_group.setLayout(feature_layout)
        layout.addWidget(feature_group)
        
        # Appearance settings
        appearance_group = QGroupBox("Görünüm Ayarları")
        appearance_layout = QGridLayout()
        
        theme_label = QLabel("Tema:")
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Koyu", "Açık"])
        self.theme_combo.setCurrentText("Koyu" if self.settings["dark_mode"] else "Açık")
        
        font_label = QLabel("Yazı Boyutu:")
        self.font_combo = QComboBox()
        self.font_combo.addItems(["Küçük", "Orta", "Büyük"])
        self.font_combo.setCurrentText(self.settings["font_size"].capitalize())
        
        appearance_layout.addWidget(theme_label, 0, 0)
        appearance_layout.addWidget(self.theme_combo, 0, 1)
        appearance_layout.addWidget(font_label, 1, 0)
        appearance_layout.addWidget(self.font_combo, 1, 1)
        
        appearance_group.setLayout(appearance_layout)
        layout.addWidget(appearance_group)
        
        # Language settings
        language_group = QGroupBox("Dil Ayarları")
        language_layout = QGridLayout()
        
        lang_label = QLabel("Dil:")
        self.lang_combo = QComboBox()
        self.lang_combo.addItems(["Türkçe", "English"])
        self.lang_combo.setCurrentText("Türkçe" if self.settings["language"] == "tr" else "English")
        
        language_layout.addWidget(lang_label, 0, 0)
        language_layout.addWidget(self.lang_combo, 0, 1)
        
        language_group.setLayout(language_layout)
        layout.addWidget(language_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        save_button = AnimatedButton("Kaydet")
        save_button.clicked.connect(self.save_settings)
        
        cancel_button = AnimatedButton("İptal")
        cancel_button.clicked.connect(self.reject)
        
        button_layout.addWidget(save_button)
        button_layout.addWidget(cancel_button)
        
        layout.addLayout(button_layout)
    
    def save_settings(self):
        """Save the current settings"""
        self.settings.update({
            "camera_index": self.camera_combo.currentIndex(),
            "confidence_threshold": self.conf_spin.value(),
            "sound_enabled": self.sound_check.isChecked(),
            "video_enabled": self.video_check.isChecked(),
            "dark_mode": self.theme_combo.currentText() == "Koyu",
            "font_size": self.font_combo.currentText().lower(),
            "language": "tr" if self.lang_combo.currentText() == "Türkçe" else "en",
            "auto_save_screenshots": self.auto_screenshot_check.isChecked(),
            "recording_enabled": self.recording_check.isChecked()
        })
        
        save_settings(self.settings)
        self.accept() 