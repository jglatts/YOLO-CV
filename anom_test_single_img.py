"""
Single Image Anomaly Detection Script using Patchcore
NOTE:
    Run in a normal terminal. Provide the image filename as a command line argument.
"""

import sys
import cv2
import numpy as np
from anomalib.data import PredictDataset
from anomalib.engine import Engine
from anomalib.models import Patchcore

# Threshold for deciding if an image is BAD
IMAGE_SCORE_THRESHOLD = 0.5  # tweak this based on your dataset

# Optional: minimum contour radius for visualization
MIN_RADIUS = 3


def analyzePrediction(pred, i, show_heatmap=True):
    """
    Analyze a single prediction, decide GOOD vs BAD, and visualize.
    """
    image_path = pred.image_path
    if isinstance(image_path, (list, tuple)):
        image_path = image_path[0]

    # Compute image anomaly score from anomaly_map
    anomaly_map = pred.anomaly_map.squeeze().cpu().numpy()
    image_score = float(np.max(anomaly_map))  # max value as image score
    # Alternative: np.mean(anomaly_map)

    # Decide if GOOD or BAD
    if image_score > IMAGE_SCORE_THRESHOLD:
        status = "BAD"
    else:
        status = "GOOD"

    print(f"\nImage {image_path} is {status} (score: {image_score:.4f})\n")

    # Load image for visualization
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Warning: failed to load image: {image_path}")
        return

    if show_heatmap:
        # Normalize heatmap
        heatmap = (255 * (anomaly_map - anomaly_map.min()) /
                   (anomaly_map.max() - anomaly_map.min() + 1e-8)).astype(np.uint8)
        heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

        # Threshold for contour visualization
        _, thresh = cv2.threshold(heatmap_resized, 60, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        output_image = image.copy()
        for cnt in contours:
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            if radius > MIN_RADIUS:
                center = (int(x), int(y))
                radius = int(radius)
                cv2.circle(output_image, center, radius, (0, 0, 255), 2)

        # Optional heatmap overlay
        heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(output_image, 0.7, heatmap_color, 0.3, 0)

        cv2.imshow(f"Image {i} - {status}", overlay)
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()


def testEngineSingleImage(image_filename):
    """
    Run anomaly detection on a single image using a trained Patchcore model.
    """
    # Load trained model
    model = Patchcore()
    engine = Engine()

    # PredictDataset for a single image
    dataset = PredictDataset(
        path="./test_images/" + image_filename,
        image_size=(512, 512)
    )

    # Checkpoint path
    ckpt_path = r"C:\Users\jglatts\Documents\Z-Axis\YOLO-CV\results\Patchcore\zfill_dataset\latest\weights\lightning\model.ckpt"

    # Run prediction
    predictions = engine.predict(
        model=model,
        dataset=dataset,
        ckpt_path=ckpt_path,
    )

    # Analyze results
    if predictions:
        for i, pred in enumerate(predictions):
            analyzePrediction(pred, i)
    else:
        print(f"\nImage {image_filename} is GOOD (no anomalies detected)\n")


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
    for img in images3:
        print("testing " + img)
        testEngineSingleImage(img)

def testWithCAM():
    pass

if __name__ == "__main__":
    if len(sys.argv) < 2:
        loopAndTest()
    else:
        testEngineSingleImage(sys.argv[1])
