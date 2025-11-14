import sys
import cv2
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel,
    QPushButton, QMessageBox, QVBoxLayout, QGroupBox,
    QRadioButton, QButtonGroup,
)
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt, QThread, pyqtSignal


class Driver():
    def __init__(self):
        pass

    def run(self):
        app = QApplication(sys.argv)
        window = MainWindow()
        window.show()
        app.exec()


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

        # Set current image
        self.qt_image = self.qt_image_rgb if self.useRGB else self.qt_image_gray

    def run(self):
        self.running = True
        while self.running:
            ret, self.frame = self.cap.read()
            if not ret:
                continue

            self.extractFrame()
            self.frame_ready.emit(self.qt_image)
            self.msleep(30)  # small delay (~30 FPS)

    def stop(self):
        self.running = False
        self.quit()
        self.wait()
        self.cap.release()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.camera_thread = None
        self.setUI()

    def setUI(self):
        self.setWindowTitle("Anomaly Detection Demo")
        self.setFixedSize(800, 600)

        central = QWidget()
        self.setCentralWidget(central)
        self.main_layout = QVBoxLayout(central)
        self.main_layout.setContentsMargins(10, 10, 10, 10)
        self.main_layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)

        # Controls group
        group_box = QGroupBox("Controls")
        group_layout = QVBoxLayout()
        group_box.setLayout(group_layout)

        self.button = QPushButton("Info")
        self.button.clicked.connect(self.testButton_Callback)
        group_layout.addWidget(self.button, alignment=Qt.AlignmentFlag.AlignLeft)

        self.button_cam = QPushButton("Start Live Feed")
        self.button_cam.clicked.connect(self.openCam_Callback)
        group_layout.addWidget(self.button_cam, alignment=Qt.AlignmentFlag.AlignLeft)

        self.main_layout.addWidget(group_box)

        # Video display
        self.video_label = QLabel()
        self.video_label.setFixedSize(640, 480)
        self.main_layout.addWidget(self.video_label)

    def update_frame(self, image):
        pixmap = QPixmap.fromImage(image)
        self.video_label.setPixmap(pixmap)

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


if __name__ == "__main__":
    gui_driver = Driver()
    gui_driver.run()
