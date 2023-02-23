import torch
import numpy as np
import cv2
import time

model = torch.hub.load("yolov5", "custom", path='yolov5/runs/train/exp4/weights/last.pt', source='local')

img = "data/images/circuit1.png"
results = model(img)
results.print()
print(results.xyxy)
cv2.imshow("yolo", np.squeeze(results.render()))
cv2.waitKey(0)


