import cv2
import os

x = 0
y = 0
smallestX = 20000
smallestY = 20000

avgY = 370
avgX = 370
dimensions = (avgX, avgY)

for i in range(1,49):
    fileName = (f"data/imgs/circuit{i}.png")
    print(fileName)
    img = cv2.imread(fileName)

    # shape = img.shape
    # print(shape)
    # x += shape[0]
    # y += shape[1]
    #
    # if shape[0] < smallestX:
    #     smallestX = shape[0]
    #
    # if shape[1] < smallestY:
    #     smallestY = shape[1]

    resized = cv2.resize(img, dimensions, interpolation=cv2.INTER_AREA)

    # cv2.imshow("Resized image", resized)
    # cv2.waitKey(0)
    cv2.imwrite(f"data/images/circuit{i}.png", resized)


# print('avg x', x/48, " y", y/48)
# print('smalles x', smallestX, " y", smallestY)
