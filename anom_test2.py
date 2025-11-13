'''

        Test Script to use a pretraind model on some images
        Will draw anomalies on the images with red circles
'''
import os
import cv2
import torch
import numpy as np
from anomalib.data import PredictDataset
from anomalib.engine import Engine
from anomalib.models import Patchcore

'''
    need more good images of parts for training
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃        Test metric        ┃       DataLoader 0        ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
        │        image_AUROC        │    0.3333333432674408     │
        │       image_F1Score       │    0.7272727489471436     │
        └───────────────────────────┴───────────────────────────┘
    we want image_AUROC closer to 1.0    

    F1_SCORE
        Range: 0 → 1
            1 = perfect
            0 = totally wrong
'''

def testEngineElastomer():
    # Load the trained model
    model = Patchcore()
    engine = Engine()

    # Path to test dataset
    dataset_path = "./datasets/Z-Axis/zfill/test"

    # Prepare test images (can be a folder)
    dataset = PredictDataset(path=dataset_path, 
                             image_size=(256, 256),
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
        print(f"\n\nnumber of of bad objects {len(predictions)}\n\n")
        for i, pred in enumerate(predictions):
            analyzeBadImage(pred, i)
            key = cv2.waitKey(0)
            cv2.destroyAllWindows()
            if key == 27:  # ESC to exit early
                break

    cv2.destroyAllWindows()


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



def testEngine():
    # Load the trained model
    model = Patchcore()
    engine = Engine()

    # Path to test dataset
    #dataset_path = "./datasets/MVTecAD/bottle/test"
    dataset_path = "./datasets/MVTecAD/transistor/test"

    # Prepare test images (can be a folder)
    dataset = PredictDataset(path=dataset_path, 
                             image_size=(256, 256),
                             )
    
    # Checkpoints 
    #ckpt_path = r'C:\Users\jglatts\Documents\Z-Axis\YOLO-CV\results\Patchcore\MVTecAD\bottle\latest\weights\lightning\model.ckpt'
    ckpt_path = r"C:\Users\jglatts\Documents\Z-Axis\YOLO-CV\results\Patchcore\MVTecAD\transistor\latest\weights\lightning\model.ckpt"

    # Predict Anomalies
    predictions = engine.predict(
        model=model,
        dataset=dataset,
        ckpt_path=ckpt_path,
    )

    # Show results
    if predictions is not None:
        print(f"\n\nnumber of of bad objects {len(predictions)}\n\n")
        for i, pred in enumerate(predictions):
            analyzeBadImage(pred, i)
            key = cv2.waitKey(0)
            cv2.destroyAllWindows()
            if key == 27:  # ESC to exit early
                break

    cv2.destroyAllWindows()



if __name__ == "__main__":
    testEngineElastomer()
