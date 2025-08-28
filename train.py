from ultralytics import YOLO
import roboflow

# 🔑 Connect to Roboflow with your API key
# 👉 Replace YOUR_API_KEY with the API key from your Roboflow account
rf = roboflow.Roboflow(api_key="UdfSai5ynct0Bslk2hZk")

# 📦 Load the Indian Emergency Vehicles dataset project
project = rf.workspace("ganesh-lbmbj").project("indian-emergency-vehicles-dataset")
dataset = project.version(1).download("yolov8")   # Version(1) is the latest dataset

# 🧠 Load YOLOv8 model (Nano for fast training)
model = YOLO("yolov8n.pt")

# 🎯 Train the model
model.train(
    data=dataset.location + "/data.yaml",  # Path to dataset
    epochs=50,
    imgsz=640
)

print("✅ Training complete! Model saved inside runs/detect/train/weights/")
