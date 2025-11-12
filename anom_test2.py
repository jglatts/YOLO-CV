import cv2
from anomalib.data import PredictDataset
from anomalib.engine import Engine
from anomalib.models import Patchcore

def testEngine():
    # 1. Load the trained model
    model = Patchcore()
    engine = Engine()

    ckpt_path = r'C:\Users\jglatts\Documents\Z-Axis\YOLO-CV\results\Patchcore\MVTecAD\bottle\latest\weights\lightning\model.ckpt'

    # 2. Prepare test images (can be a folder)
    dataset = PredictDataset(path="./datasets/MVTecAD/bottle/test", image_size=(256, 256))

    # 3. Predict anomalies
    predictions = engine.predict(
        model=model,
        dataset=dataset,
        ckpt_path=ckpt_path
    )

    if predictions is not None:
        for pred in predictions:
            # Handle image path
            image_path = pred.image_path
            if isinstance(image_path, (list, tuple)):
                image_path = image_path[0]

            image = cv2.imread(str(image_path))
            if image is None:
                print(f"Warning: failed to load image: {image_path}")
                continue

            # Handle anomaly map
            anomaly_map = pred.anomaly_map
            heatmap = (anomaly_map * 255).byte().cpu().numpy()
            heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

            overlay = cv2.addWeighted(image, 0.7, heatmap_color, 0.3, 0)

            cv2.imshow("Anomaly Detection", overlay)
            cv2.waitKey(0)

    cv2.destroyAllWindows()



if __name__ == "__main__":
    testEngine()
