import sys
import cv2
import numpy as np
import tempfile
import os
from anomalib.data import PredictDataset
from anomalib.engine import Engine
from anomalib.models import Patchcore

IMAGE_SCORE_THRESHOLD = 0.5
MIN_RADIUS = 3

def analyzePrediction(pred, i, show_heatmap=True, frame_override=None):
    """
    Analyze Patchcore prediction and visualize.
    If frame_override is provided, it will draw on that image instead of loading from disk.
    """
    # anomaly map
    anomaly_map = pred.anomaly_map.squeeze().cpu().numpy()
    image_score = float(np.max(anomaly_map))

    status = "BAD" if image_score > IMAGE_SCORE_THRESHOLD else "GOOD"
    print(f"[CAM] Frame {i}: {status} (score={image_score:.4f})")

    # choose the image (frame_override is a BGR OpenCV frame)
    if frame_override is not None:
        image = frame_override.copy()
    else:
        image_path = pred.image_path[0] if isinstance(pred.image_path, (list, tuple)) else pred.image_path
        image = cv2.imread(str(image_path))

    if not show_heatmap:
        cv2.imshow("Output", image)
        return status

    # create heatmap
    heatmap = (255 * (anomaly_map - anomaly_map.min()) /
               (anomaly_map.max() - anomaly_map.min() + 1e-8)).astype(np.uint8)
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

    _, thresh = cv2.threshold(heatmap, 60, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output_image = image.copy()
    if status == "BAD":
        for cnt in contours:
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            if radius > MIN_RADIUS:
                cv2.circle(output_image, (int(x), int(y)), int(radius), (0, 0, 255), 2)

    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(output_image, 0.7, heatmap_color, 0.3, 0)

    cv2.imshow(f"Heatmap {status}", overlay)
    cv2.imshow(f"Live {status}", image)

    return status


def predict_frame(frame):
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


def runUSBcam():
    """
    Live anomaly detection using USB camera and Patchcore.
    """
    print("Starting USB camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open USB camera")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame capture failed")
            break

        # Run Patchcore on this frame
        preds = predict_frame(frame)

        if preds:
            for i, pred in enumerate(preds):
                analyzePrediction(pred, i, show_heatmap=True, frame_override=frame)
        else:
            cv2.imshow("Live GOOD", frame)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Original single-image function still works
def testEngineSingleImage(image_filename):
    model = Patchcore()
    engine = Engine()

    dataset = PredictDataset(
        path="./test_images/" + image_filename,
        image_size=(512, 512)
    )

    ckpt_path = r"C:\Users\jglatts\Documents\Z-Axis\YOLO-CV\results\Patchcore\zfill_dataset\latest\weights\lightning\model.ckpt"

    predictions = engine.predict(
        model=model,
        dataset=dataset,
        ckpt_path=ckpt_path,
    )

    if predictions:
        for i, pred in enumerate(predictions):
            analyzePrediction(pred, i)
    else:
        print(f"\nImage {image_filename} is GOOD\n")


def loopAndTest():
    images = ["test_1.png",
              "test_2.png",
              "test_3.png",
              "test_4.png",
              "test_5.png",
              ]
    images2 = ["bad_6.png",
            "bad_7.png",
            "bad_8.png",
            "bad_9.png",
            "bad_10.png",
            ]
    images3 = ["good_1.png",
            "good_2.png",
            "good_3.png",
            "good_4.png",
            "good_5.png",
            ]
    for img in images:
        print("testing " + img)
        testEngineSingleImage(img)
        if cv2.waitKey(0) & 0xFF == ord('q'):
           break
        cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "cam":
        runUSBcam()
    elif len(sys.argv) > 1:
        testEngineSingleImage(sys.argv[1])
    else:
        loopAndTest()