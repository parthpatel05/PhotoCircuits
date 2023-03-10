from ultralytics import YOLO
import time
import cv2
# Load a model
model = YOLO()


model = YOLO("runs\detect/train/weights/best.pt")
img = "data/images/circuit1.png"
results = model.predict(source=img)  # can also put ,save=True
results = results[0].boxes
boxes = results.xyxy.tolist() # this holds the bounding box coordinates
classes = results.cls.tolist() # this holds the classes for the boxes
print(boxes)
print(classes)

image = cv2.imread(img)
for box in boxes:
    topCorner = [int(box[0]),int(box[1])]
    bottomCorner = [int(box[2]),int(box[3])]
    # image = cv2.circle(image, topCorner, radius=5, color=(0, 0, 255), thickness=-1)
    image = cv2.rectangle(image, topCorner, bottomCorner, color=(0, 0, 255), thickness=-1)

cv2.imshow("yolo",image)
cv2.imwrite("fromv8.png", image)
cv2.waitKey(0)

