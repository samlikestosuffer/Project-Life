from ultralytics import YOLO

# Load the trained model
model = YOLO("runs/detect/train/weights/best.pt")

# Run detection on a video
results = model.predict(source="traffic.mp4", conf=0.1, show=True)

# The output video will be saved in runs/detect/predict by default
