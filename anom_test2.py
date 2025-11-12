'''

        Test Script to use a pretraind model on some images
        Will draw anomalies on the images with red circles
'''
import cv2
import torch
from anomalib.data import PredictDataset
from anomalib.engine import Engine
from anomalib.models import Patchcore
import os

def analyzeBadImage(pred, i):
    image_path = pred.image_path
    if isinstance(image_path, (list, tuple)):
        image_path = image_path[0]

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Warning: failed to load image: {image_path}")
        return

    print(f"\nimage {image_path} is BAD\n")

    # Get anomaly map tensor
    anomaly_map = pred.anomaly_map  # PyTorch tensor

    # Convert anomaly map to numpy array, scale 0-255
    heatmap = (anomaly_map * 255).clamp(0, 255).byte().cpu().numpy()
    if heatmap.ndim != 2:
        heatmap = heatmap[0]

    # Resize anomaly map to original image size
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

    # Threshold to find anomaly regions
    _, thresh = cv2.threshold(heatmap_resized, 128, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Copy original image to draw on
    output_image = image.copy()

    # Draw red circles around each anomaly
    for cnt in contours:
        # Get minimal enclosing circle
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        if radius > 5:  # ignore tiny noise
            center = (int(x), int(y))
            radius = int(radius)
            cv2.circle(output_image, center, radius, (0, 0, 255), 2)  # red circle

    # Show the image
    cv2.imshow(f"img {i}", output_image)


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
    testEngine()
