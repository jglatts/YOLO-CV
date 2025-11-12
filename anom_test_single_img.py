import os
import sys
import cv2
import torch
import numpy as np
from anomalib.data import PredictDataset
from anomalib.engine import Engine
from anomalib.models import Patchcore


def analyzeBadImage(pred, i):
    image_path = pred.image_path
    if isinstance(image_path, (list, tuple)):
        image_path = image_path[0]

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Warning: failed to load image: {image_path}")
        return

    print(f"\nimage {image_path} is BAD #{i}\n")

    anomaly_map = pred.anomaly_map  # PyTorch tensor

    # Normalize heatmap dynamically
    heatmap = anomaly_map.squeeze().cpu().numpy()
    heatmap = (255 * (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)).astype(np.uint8)

    # Resize to original image
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

    # Threshold
    _, thresh = cv2.threshold(heatmap_resized, 60, 255, cv2.THRESH_BINARY)  # tweak threshold

    # Contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output_image = image.copy()
    for cnt in contours:
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        if radius > 3:  # smaller than default for elastomer defects
            center = (int(x), int(y))
            radius = int(radius)
            cv2.circle(output_image, center, radius, (0, 0, 255), 2)

    # Optional overlay
    #heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    #overlay = cv2.addWeighted(output_image, 0.7, heatmap_color, 0.3, 0)
    overlay = output_image

    cv2.imshow(f"img {i}", overlay)


def testEngineSingleImage(image_path):
    # Load the trained model
    model = Patchcore()
    engine = Engine()

    # Load the dataset (single image)
    image_path = "./test_images/" + image_path
    dataset = PredictDataset(
        path=image_path,      
        image_size=(512, 512),
    )
    
    # Checkpoints 
    ckpt_path = r"C:\Users\jglatts\Documents\Z-Axis\YOLO-CV\results\Patchcore\zfill_dataset\latest\weights\lightning\model.ckpt"

    # Predict Anomalies
    predictions = engine.predict(
        model=model,
        dataset=dataset,
        ckpt_path=ckpt_path,
    )

    # Show results
    if predictions is not None:
        for i, pred in enumerate(predictions):
            analyzeBadImage(pred, i)
            key = cv2.waitKey(0)
            cv2.destroyAllWindows()
            if key == 27:  # ESC to exit early
                break
    else:
        print("\n\nGOOD PART!\n\n")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    if (len(sys.argv) < 2):
       print("error please supply image file name!")
    else:
        testEngineSingleImage(str(sys.argv[1]))