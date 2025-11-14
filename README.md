# YOLO-CV

YOLO-CV is a concept project for experimenting with real-time computer vision using **Ultralytics YOLO (v8/v11)** and **Anomalib** anomaly-detection models. It includes a lightweight PyQt6 interface for running inference on images, URLs, or a live USB camera feed.

---

## Features

- Run inference using:
  - **YOLO models** (object detection)
  - **Anomalib models** (PatchCore, PADIM, etc.)
- Input sources:
  - Local image files
  - Image URLs
  - Live USB camera
- Real-time visualization in a PyQt6 GUI
- Bounding boxes, anomaly maps, and status overlays
- Simple, extendable architecture for experimentation

---

## UI

![UI](https://raw.githubusercontent.com/jglatts/YOLO-CV/refs/heads/master/sys-images/ui.png)

---

## Installation

```bash
git clone https://github.com/jglatts/YOLO-CV.git
cd YOLO-CV
