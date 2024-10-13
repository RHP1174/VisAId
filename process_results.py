from ultralytics import YOLO
import cv2
import time

def process_results():
    model = YOLO("yolo11n.pt")
    cap = cv2.VideoCapture(0)
    duration = 15
    start_time = time.time()
    object_counts = {}
    tracked_ids = set()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        results = model.track(frame, conf=0.69, persist=True)

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            clss = results[0].boxes.cls.cpu().tolist()

            for box, track_id, cls in zip(boxes, track_ids, clss):
                if track_id not in tracked_ids:
                    class_name = model.names[int(cls)]
                    object_counts[class_name] = object_counts.get(class_name, 0) + 1
                    tracked_ids.add(track_id)

                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                label = f'{model.names[int(cls)]} ID:{track_id}'
                cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        cv2.imshow("Object Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q') or time.time() - start_time > duration:
            break

    cap.release()
    cv2.destroyAllWindows()

    return object_counts