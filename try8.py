from ultralytics import YOLO
import time
import cv2
# Load a model
model = YOLO()


model = YOLO("runs\detect/train8/weights/best.pt")
img = "data/images/circuit1.png"
results = model.predict(source=img,save=True)  # can also put save=true
results = results[0].boxes
boxes = results.xyxy.tolist() # this holds the bounding box coordinates
classes = results.cls.tolist() # this holds the classes for the boxes
print(boxes)
print(classes)


# image = cv2.circle(img, (results[0],results[1]), radius=0, color=(0, 0, 255), thickness=-1)
# # cv2.imshow("yolo",image)
# # cv2.waitKey(0)


# todo start the training, and the find how to get the xy coordinates