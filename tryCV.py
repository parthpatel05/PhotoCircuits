import cv2
import numpy as np
img = cv2.imread("inn.jpg")

img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Threshold of blue in HSV space
lower_blue = np.array([0,0,0])
upper_blue = np.array([360,255,80])

# preparing the mask to overlay
img = cv2.inRange(img, lower_blue, upper_blue)

# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.GaussianBlur(img, (7,7), 1)

img = cv2.Canny(img, 200, 200)




cv2.imshow("out", img)
cv2.waitKey(0)
#cv2.imwrite("out7.jpg", img)
