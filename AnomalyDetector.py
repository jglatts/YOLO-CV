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
            return self.status

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
