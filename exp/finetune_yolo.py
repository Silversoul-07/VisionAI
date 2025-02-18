from ultralytics import YOLO

model = YOLO("yolo11x.pt")

data_yaml = "path/to/data.yaml"

# Fine-tune the model (transfer learning)
model.train(data=data_yaml, epochs=10, imgsz=640, batch=16)