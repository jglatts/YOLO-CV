from ultralytics import YOLO
import cv2
import urllib.request
import numpy as np

'''
    need to train YOLO on custom data
        model = YOLO("yolov8n.pt")  # start with small model
        model.train(data="your_dataset.yaml", epochs=50)
'''

def getImgData(url):
    req = urllib.request.Request(
        url=url, 
        headers={"User-Agent": "Mozilla/5.0"}
    )
    resp = urllib.request.urlopen(req)
    image_array = np.asarray(bytearray(resp.read()), dtype=np.uint8)
    img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return img


def test():
    # Load a pretrained YOLO model
    model = YOLO("yolov8x.pt")  

    #url = "https://ultralytics.com/images/bus.jpg"
    url = "https://www.zaxisconnector.com/wp-content/uploads/2024/11/file-12.png"

    img = getImgData(url)

    # Perform detection on the image
    results = model.predict(source=img, imgsz=640)

    print("len of results is " + str(len(results)))

    # Loop through results
    for result in results:
        plotted_img = result.plot()
        cv2.imshow("Test", plotted_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    test()
