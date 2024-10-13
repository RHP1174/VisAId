from ultralytics import YOLO

def process_results(model_path, source=0):
    # Load the YOLO model
    model = YOLO(model_path)

    # Run the model
    res = model(source, show=True)

    # List to store detected objects and confidence scores
    detected_objects = []

    # Process the results
    for results in res:
        boxes = results.boxes
        names = results.names
        conf = results.conf

        for box, name, score in zip(boxes, names, conf):
            detected_objects.append((name, score))  # Store name and confidence score

    return detected_objects

if __name__ == "__main__":
    # Specify the model path and source (0 for webcam)
    detected = process_results("yolo11n.pt", source=0)

    # Print or save the results
    print(detected)
