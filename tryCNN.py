import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2

# model = torch.hub.load('ultralytics/yolov5', "yolov5s")
# img = "img.jpg"
#
# results = model(img)
# results.print()
# print(results.xyxy)
# # cv2.imshow("yolo", np.squeeze(results.render()))
# # cv2.waitKey(0)

model = torch.hub.load("ultralytics/yolov5", "custom", path='yolov5/runs/train/exp4/weights/last.pt', force_reload=True)

img = "data/images/circuit3.png"
results = model(img)
results.print()
print(results.xyxy)
cv2.imshow("yolo", np.squeeze(results.render()))
cv2.waitKey(0)
