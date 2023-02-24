from ultralytics import YOLO
import time
# Load a model
model = YOLO()

if __name__ == '__main__':

    # Use the model
    results = model.train(data="dataMnist.yaml", epochs=5)  # train the model
    # success = YOLO("yolov8n.pt").export(format="onnx")
