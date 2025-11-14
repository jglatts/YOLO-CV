import cv2
import sys
import tempfile
import os
import numpy as np
from datetime import datetime
from anomalib.data import PredictDataset
from anomalib.engine import Engine
from anomalib.models import Patchcore
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel,
    QPushButton, QMessageBox, QVBoxLayout, QGroupBox,
    QRadioButton, QButtonGroup, QHBoxLayout
)
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal


# Worker thread to capture frames
class CameraThread(QThread):
    frame_ready = pyqtSignal(QImage)

    def __init__(self, camera_index=0):
        super().__init__()
        self.cap = cv2.VideoCapture(camera_index)
        self.running = False  
        self.useRGB = False

    def extractFrame(self):
        # Convert to grayscale
        gray_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)
        self.process_frame = rgb_frame.copy() # keep a copy for processing
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        self.qt_image_gray = QImage(rgb_frame.data, w, 
                               h, bytes_per_line, 
                               QImage.Format.Format_RGB888)

        # Color version
        rgb_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)  # real RGB
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        self.qt_image_rgb = QImage(rgb_frame.data, w, 
                                   h, bytes_per_line, 
                                   QImage.Format.Format_RGB888)


    def run(self):
        self.running = True
        while self.running:
            ret, self.frame = self.cap.read()
            if not ret:
                continue

            self.extractFrame()
            self.frame_ready.emit(self.qt_image_rgb)
            self.msleep(30)  # small delay (~30 FPS)

    def stop(self):
        self.running = False
        self.quit()
        self.wait()
        self.cap.release()
