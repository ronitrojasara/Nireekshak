from ultralytics import YOLO

model = YOLO("./vision model/your_model.pt")
model.predict(source="0", show=True)
