from ultralytics import YOLO

model = YOLO("yolo11n.pt")
res = model(0, show = True)

for results in res:
  boxes = results.boxes
  classes = results.names
  conf = results.conf