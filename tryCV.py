import cv2
import numpy as np
# img = cv2.imread("testData/circuit10.png")
img = cv2.imread("inn3.jpg")
imgCont = img.copy()

# x = 110
# y = 44
# img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# print(img[x,y])
# cv2.circle(img, (x,y), radius=0, color=(0, 0, 0), thickness=2)
# cv2.imshow("mask", img)
# cv2.waitKey(0)

# Threshold of blue in HSV space
lower_blk = np.array([0,0,0])
upper_blk = np.array([170,260,260])

# preparing the mask to overlay
img = cv2.inRange(img, lower_blk, upper_blk)

cv2.imshow("mask", img)



# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.GaussianBlur(img, (7,7), 1)

# cv2.imshow("blur", img)
img = cv2.Canny(img, 200, 200)

cnts, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
counter = 0
countours = []
for cnt in cnts:
    counter += 1
    # print( cnt)
    area = cv2.contourArea(cnt)
    if area > 100:
        cv2.drawContours(imgCont, cnt, -1, (0,0,255),-1)
        cv2.fillPoly(imgCont, pts=[cnt], color=(0, 0, 255))
        peri = cv2.arcLength(cnt,True)
        approx = cv2.approxPolyDP(cnt, .02*peri, True)
        objCor = len(approx)
        x , y , w, h = cv2.boundingRect(approx)
        # countours.append(cnt)
        countours.append(cnt.tolist())

print(countours)

# print("couter", counter)
cv2.imshow("out", img)
cv2.imshow("out1", imgCont)
cv2.waitKey(0)
