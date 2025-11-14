import sys
import cv2
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
from AnomalyDetector import AnomalyDetector

class DetectionThread(QThread):
    result_ready = pyqtSignal(np.ndarray)  # emits processed frame

    def __init__(self):
        super().__init__()
        self.detector = AnomalyDetector()
        self.frame = None
        self.running = False

    def set_frame(self, frame):
        self.frame = frame

    def run(self):
        self.running = True
        while self.running:
            if self.frame is not None:
                result = self.detector.predict(self.frame)
                self.result_ready.emit(result)
                self.frame = None
            self.msleep(10)

    def stop(self):
        self.running = False
        self.quit()
        self.wait()