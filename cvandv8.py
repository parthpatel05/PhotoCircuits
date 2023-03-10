import cv2
import numpy as np
from ultralytics import YOLO
import time
import cv2
import random
# Load a model
model = YOLO()
model = YOLO("runs\detect/train/weights/best.pt")
# 3not all nodes,6 too many noise,18 all red y/cuz there big box around,29 not all nodes detected, 31 still the no good area problem
# problem solved but check   10 same node seperated,14 nodes, 16 nodes and noise, 21 too many nodes for one, 25 nodes think shape no closed,  28 nodes same as 25,
# skinny lines and grey lines r problemo
image = "data/images/circuit31.png"
# image = "un1.jpg"
# image = "testData/circuit6.png"

results = model.predict(source=image)  # can also put ,save=True
results = results[0].boxes
boxes = results.xyxy.tolist() # this holds the bounding box coordinates
classes = results.cls.tolist() # this holds the classes for the boxes
print(boxes)
print(classes)

img = cv2.imread(image)
imgOrg = img.copy()
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.GaussianBlur(img, (7,7), 1)
(thresh, blackAndWhiteImage) = cv2.threshold(img, 245, 255, cv2.THRESH_BINARY)
cv2.imshow("blk",blackAndWhiteImage)
img = cv2.cvtColor(blackAndWhiteImage, cv2.COLOR_GRAY2RGB)
# cv2.imshow("cor",img)

for box in boxes:
    topCorner = [int(box[0]),int(box[1])]
    bottomCorner = [int(box[2]),int(box[3])]

    cv2.rectangle(img, topCorner, bottomCorner, color=(0, 0, 255), thickness=-1)
    cv2.rectangle(imgOrg, topCorner, bottomCorner, color=(0, 0, 255), thickness=1)

# cv2.imshow("yolo",imgOrg)
# cv2.waitKey(0)

img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.imshow("hsv",img)
# Threshold of blk in HSV space
lower_blk = np.array([0,0,0])
upper_blk = np.array([360,255,80])
# upper_blk = np.array([170,260,260])

# mask, blur, canny
img = cv2.inRange(img, lower_blk, upper_blk)
cv2.imshow("mask", img)
img = cv2.GaussianBlur(img, (7,7), 1)
cv2.imshow("blur", img)
img = cv2.Canny(img, 200, 200)

cnts, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
counter = 0
print(len(cnts))

countours = []
for cnt in cnts:
    area = cv2.contourArea(cnt)
    peri = cv2.arcLength(cnt, True)
    print(counter, area, int(peri))
    if area > 50 or peri > 500:
        color = (random.randint(0, 255),random.randint(0, 255),random.randint(0, 255))
        cv2.putText(imgOrg, f'n:{counter}', (int(cnt[0][0][0]), int(cnt[0][0][1])), cv2.FONT_HERSHEY_SIMPLEX, .7, color)
        # print(counter, colors[counter])
        cv2.fillPoly(imgOrg, pts=[cnt], color=color)
        # peri = cv2.arcLength(cnt,True)
        # approx = cv2.approxPolyDP(cnt, .02*peri, True)
        # objCor = len(approx)
        # x , y , w, h = cv2.boundingRect(approx)
        counter += 1
        countours.append(cnt.tolist())
    else:
        cv2.fillPoly(imgOrg, pts=[cnt], color=(128, 0, 128))


    # color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    # cv2.putText(imgOrg, f'n:{counter}', (int(cnt[0][0][0]), int(cnt[0][0][1])), cv2.FONT_HERSHEY_SIMPLEX, .7,
    #             color)
    # cv2.fillPoly(imgOrg, pts=[cnt], color=color)
    # counter += 1
    # countours.append(cnt.tolist())

cv2.imshow("canny", img)
cv2.imshow("out1", imgOrg)
# cv2.waitKey(0)

conectionDict = {}
# loop through the features
for featureInd in range(len(classes)):
    coor = boxes[featureInd]
    print(coor)
    if classes[featureInd] == 0:
        print("voltagh", featureInd)
        cv2.putText(imgOrg, f"f{featureInd}", (int(coor[0]), int(coor[3])), cv2.FONT_HERSHEY_SIMPLEX, .7, 255)
    elif classes[featureInd] == 1:
        print("res", featureInd)
        cv2.putText(imgOrg, f"f{featureInd}", (int(coor[0]), int(coor[3])), cv2.FONT_HERSHEY_SIMPLEX, .7, 255)
    elif classes[featureInd] == 3:
        print("curr", featureInd)
        cv2.putText(imgOrg, f"f{featureInd}", (int(coor[0]), int(coor[3])), cv2.FONT_HERSHEY_SIMPLEX, .7, 255)

    # then loop through the nodes
    for node in range(len(countours)):
        cnt = countours[node]
        print(cnt)
        # point in this format [[x y]]
        for point in cnt:
            # if a point in the node is within the x and within the y
            if int(coor[0])-5 < point[0][0] and point[0][0] < int(coor[2])+5 and int(coor[1])-5 < point[0][1] and point[0][1] < int(coor[3])+5:
                print('feature', featureInd, 'node', node)
                if featureInd in conectionDict.keys() and node not in conectionDict[featureInd]:
                    conectionDict[featureInd].append(node)
                elif featureInd not in conectionDict.keys():
                    conectionDict[featureInd] = [node,]


print(conectionDict)


cv2.imshow("out1", imgOrg)
cv2.waitKey(0)
# todo line detection not the best always - why some clearly big area lines not give right area
# todo need a way to determine the closest line to a feature
# todo gets confused when more symbols on the circuit

