import cv2
import numpy as np
file = "in7.png"
img = cv2.imread(file)

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow("out", img)
cv2.waitKey(0)
cv2.imwrite(file, img)
