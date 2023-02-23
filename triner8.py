from ultralytics import YOLO
import time
# Load a model
model = YOLO()

if __name__ == '__main__':

    # Use the model
    results = model.train(data="data2.yaml", epochs=100)  # train the model
    success = YOLO("yolov8n.pt").export(format="onnx")
