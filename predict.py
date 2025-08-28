from ultralytics import YOLO

# Load your trained model
model = YOLO("runs/detect/train/weights/best.pt")

# Run prediction on video or image
results = model.predict(
    source="traffic.mp4",  # change to "ambulance.jpg" for image
    conf=0.25,
    show=True
)

print("✅ Prediction completed. Check 'runs/detect/predict' folder for results.")
