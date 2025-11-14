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
from DetectionThread import DetectionThread
from CameraThread import CameraThread

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.camera_thread = None
        self.detection_thread = DetectionThread()
        self.setUI()

    def setUI(self):
        self.setWindowTitle("Anomaly Detection Demo")
        self.setFixedSize(800, 600)

        # ML model init 
        self.detection_thread.result_ready.connect(self.update_detection_result)
        self.detection_thread.start()

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        self.main_layout = QHBoxLayout(central)  # horizontal layout
        self.main_layout.setContentsMargins(10, 10, 10, 10)
        self.main_layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)

        # Video display (stacked vertically)
        self.video_layout = QVBoxLayout()  # vertical stack for video
        self.raw_video_label = QLabel("Raw Feed")
        self.raw_video_label.setFixedSize(640, 240)
        self.processed_video_label = QLabel("Processed Feed")
        self.processed_video_label.setFixedSize(640, 240)

        self.video_layout.addWidget(self.raw_video_label)
        self.video_layout.addWidget(self.processed_video_label)

        self.main_layout.addLayout(self.video_layout)

        # Controls layout on the right
        controls_layout = QVBoxLayout()

        # Info & camera buttons
        group_box = QGroupBox("Controls")
        group_layout = QVBoxLayout()
        group_box.setLayout(group_layout)

        self.button = QPushButton("Info")
        self.button.clicked.connect(self.testButton_Callback)
        group_layout.addWidget(self.button, alignment=Qt.AlignmentFlag.AlignLeft)

        self.button_cam = QPushButton("Start Live Feed")
        self.button_cam.clicked.connect(self.openCam_Callback)
        group_layout.addWidget(self.button_cam, alignment=Qt.AlignmentFlag.AlignLeft)

        controls_layout.addWidget(group_box)

        # Color/Gray radio buttons
        color_group_box = QGroupBox("Display Mode")
        color_layout = QVBoxLayout()
        color_group_box.setLayout(color_layout)

        self.radio_gray = QRadioButton("Grayscale")
        self.radio_gray.setChecked(True)  # default
        self.radio_color = QRadioButton("Color")
        color_layout.addWidget(self.radio_gray)
        color_layout.addWidget(self.radio_color)

        # Group them for easy checking
        self.display_mode_group = QButtonGroup()
        self.display_mode_group.addButton(self.radio_gray)
        self.display_mode_group.addButton(self.radio_color)

        controls_layout.addWidget(color_group_box)

        # Add stretch to push controls to the top
        controls_layout.addStretch()

        # Add the controls layout to the main layout
        self.main_layout.addLayout(controls_layout)

        # Anomaly detection controls (as a separate group box)
        detection_group_box = QGroupBox("Anomaly Detection")
        detection_layout = QVBoxLayout()
        detection_group_box.setLayout(detection_layout)

        self.radio_anomalib_zacc = QRadioButton("Elastomers")
        self.radio_anomalib_zacc.setChecked(True)  # default
        detection_layout.addWidget(self.radio_anomalib_zacc)

        self.main_layout.addWidget(detection_group_box)

    def update_detection_result(self, frame):
        # clean this up later
        try:
            display_frame = self.detection_thread.detector.overlay.copy()
            msg = "analyzed frame"
        except:
            display_frame = frame
            msg = "raw frame"

        # Add timestamp and message as multiple lines
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines = [timestamp, msg, self.detection_thread.detector.status]  # each line separately
        y0 = 30       # starting y position
        dy = 30       # line spacing in pixels
        for i, line in enumerate(lines):
            y = y0 + i * dy
            cv2.putText(
                display_frame,
                line,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,          # font scale
                (0, 255, 0),  # color (green)
                2,          # thickness
                cv2.LINE_AA
            )

        # Convert BGR (OpenCV) to RGB (Qt)
        rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)

        # Convert to QImage
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

        # Display in QLabel
        self.processed_video_label.setPixmap(QPixmap.fromImage(qimg).scaled(
            self.processed_video_label.width(),
            self.processed_video_label.height(),
            Qt.AspectRatioMode.KeepAspectRatio))

    def update_frame(self, image):
        # Show live feed in raw video label
        if self.radio_gray.isChecked():
            display_image = self.camera_thread.qt_image_gray
        else:
            display_image = self.camera_thread.qt_image_rgb

        self.raw_video_label.setPixmap(QPixmap.fromImage(display_image))

        # Set frame for detection
        if self.camera_thread.process_frame is not None:
            self.detection_thread.set_frame(self.camera_thread.process_frame)

    def closeEvent(self, event):
        if self.camera_thread:
            self.camera_thread.stop()
        super().closeEvent(event)

    def testButton_Callback(self):
        self.show_popup("testing anomaly detection!", "JDG")

    def openCam_Callback(self):
        if self.camera_thread is None:  # only create once
            self.camera_thread = CameraThread()
            self.camera_thread.frame_ready.connect(self.update_frame)
            self.camera_thread.start()
        else:
            self.show_popup("Camera is already running!", "JDG")

    def show_popup(self, msg="", title=""):
        msgBox = QMessageBox()
        msgBox.setWindowTitle(title)
        msgBox.setText(msg)
        msgBox.exec()


class Driver():
    def __init__(self):
        pass

    def run(self):
        app = QApplication(sys.argv)
        window = MainWindow()
        window.show()
        app.exec()


# Driver code
if __name__ == "__main__":
    gui_driver = Driver()
    gui_driver.run()
