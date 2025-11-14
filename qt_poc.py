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


class Driver():
    def __init__(self):
        pass

    def run(self):
        app = QApplication(sys.argv)
        window = MainWindow()
        window.show()
        app.exec()


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


class AnomalyDetector():
    def __init__(self):
        self.IMAGE_SCORE_THRESHOLD = 0.5
        self.MIN_RADIUS = 3

    def analyzePrediction(self, pred, i, show_heatmap=True, frame_override=None):
        """
        Analyze Patchcore prediction 
        """
        # anomaly map
        anomaly_map = pred.anomaly_map.squeeze().cpu().numpy()
        image_score = float(np.max(anomaly_map))

        self.status = "BAD" if image_score > self.IMAGE_SCORE_THRESHOLD else "GOOD"
        print(f"[CAM] Frame {i}: {self.status} (score={image_score:.4f})")

        # choose the image (frame_override is a BGR OpenCV frame)
        if frame_override is not None:
            image = frame_override.copy()
        else:
            image_path = pred.image_path[0] if isinstance(pred.image_path, (list, tuple)) else pred.image_path
            image = cv2.imread(str(image_path))

        if not show_heatmap:
            return status

        # create heatmap
        heatmap = (255 * (anomaly_map - anomaly_map.min()) /
                   (anomaly_map.max() - anomaly_map.min() + 1e-8)).astype(np.uint8)
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

        _, thresh = cv2.threshold(heatmap, 60, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        print(f"\n\nimage {i} is {self.status}\n\n")

        output_image = image.copy()
        if self.status == "BAD":
            for cnt in contours:
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                if radius > self.MIN_RADIUS:
                    cv2.circle(output_image, (int(x), int(y)), int(radius), (0, 0, 255), 2)

        self.heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        self.overlay = cv2.addWeighted(output_image, 0.7, self.heatmap_color, 0.3, 0)
        self.frame = self.overlay

    def analyze(self, frame):
        """
        Runs Patchcore prediction on an OpenCV BGR frame.
        """
        model = Patchcore()
        engine = Engine()

        # save frame temporarily
        # could be taking up time
        temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        cv2.imwrite(temp_file.name, frame)

        dataset = PredictDataset(
            path=temp_file.name,
            image_size=(512, 512)
        )

        ckpt_path = r"C:\Users\jglatts\Documents\Z-Axis\YOLO-CV\results\Patchcore\zfill_dataset\latest\weights\lightning\model.ckpt"
    
        predictions = engine.predict(
            model=model,
            dataset=dataset,
            ckpt_path=ckpt_path,
        )

        return predictions

    def predict(self, frame):
        # Run Patchcore on this frame
        preds = self.analyze(frame)

        if preds:
            for i, pred in enumerate(preds):
                self.analyzePrediction(pred, i, show_heatmap=True, frame_override=frame)
        else:
            self.frame = frame

        return self.frame


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

        # Video display on the left
        self.video_label = QLabel()
        self.video_label.setFixedSize(640, 480)
        self.main_layout.addWidget(self.video_label)

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
        self.video_label.setPixmap(QPixmap.fromImage(qimg))

    def update_frame(self, image):
        if self.radio_gray.isChecked():
            display_image = self.camera_thread.qt_image_gray
        else:
            display_image = self.camera_thread.qt_image_rgb

        # shows live feed    
        # this is fast and overrides detection display
        #self.video_label.setPixmap(QPixmap.fromImage(display_image))

        # set frame for detection 
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




# Driver code
if __name__ == "__main__":
    gui_driver = Driver()
    gui_driver.run()
