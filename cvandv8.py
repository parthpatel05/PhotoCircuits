import cv2
import numpy as np
img = cv2.imread("inn4.jpg")
imgCont = img.copy()

img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Threshold of blue in HSV space
lower_blk = np.array([0,0,0])
upper_blk = np.array([360,255,80])

# preparing the mask to overlay
img = cv2.inRange(img, lower_blk, upper_blk)

# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.GaussianBlur(img, (7,7), 1)

img = cv2.Canny(img, 200, 200)

countours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
counter = 0
colors = [(0,0,255), (0,255,0), (255,0,0)]
for cnt in countours:
    print("------")
    # print( cnt)
    print(counter, colors[counter])
    area = cv2.contourArea(cnt)
    if area > 100:
        # cv2.drawContours(imgCont, cnt, -1, (0,0,255),-1)
        cv2.fillPoly(imgCont, pts=[cnt], color=colors[counter])
        peri = cv2.arcLength(cnt,True)
        approx = cv2.approxPolyDP(cnt, .02*peri, True)
        objCor = len(approx)
        x , y , w, h = cv2.boundingRect(approx)
    counter += 1

# print("couter", counter)

# cv2.imshow("out", img)
cv2.imshow("out1", imgCont)
cv2.waitKey(0)

# loop through the features
# then loop through the nodes
# if a point in the node is within the x and within the y