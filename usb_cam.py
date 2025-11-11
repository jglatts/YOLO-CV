from ultralytics import YOLO
import cv2

def open_camera():
    # Open the first USB camera (usually index 0)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return None
    return cap


def print_frame_details(model, frame, label_map):
    # Perform detection on the frame
    results = model.predict(source=frame, imgsz=640, verbose=False)

    # Draw results on the frame
    for result in results:
        frame_with_boxes = result.plot()
        # Extract class IDs and convert to names
        if result.boxes is not None and len(result.boxes) > 0:
            class_ids = result.boxes.cls.cpu().numpy()  # class IDs
            for class_id in class_ids:
                label = model.names[int(class_id)]  # get class name
                if (label in label_map):
                    label_map[label] += 1
                else:
                    label_map[label] = 1
                print("Detected " + str(label) + " seen " + str(label_map[label]) + " times")


    cv2.imshow('YOLO Camera', frame_with_boxes)


def test_camera():
    # Load a pretrained YOLO model
    #model = YOLO("yolov8x.pt")  
    model = YOLO("yolov8s.pt")  

    label_map = {}

    # Open the first USB camera (usually index 0)
    cap = open_camera()
    if cap == None:
        return

    while True:
        # Read a frame from the camera
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame\nExiting...")
            break

        # Display frame details with YOLO detections
        print_frame_details(model, frame, label_map)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    test_camera()
