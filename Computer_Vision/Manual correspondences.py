import cv2
from matplotlib import pyplot as plt
import numpy as np


def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        lx.append(x)
        ly.append(y)
        cv2.circle(img, (x, y), 1, (0, 0, 255), thickness=-1)
        cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (0, 0, 0), thickness=1)
        cv2.imshow("image", img)
        print(x, y)

lx = []
ly = []
rx = []
ry = []

img1 = cv2.imread("D:\IC-Msc\Computer Vision\Original Picture\FG\DSCF0165.jpg")
img2 = cv2.imread("D:\IC-Msc\Computer Vision\Original Picture\FG\DSCF0166.jpg")

imgs = np.hstack([img1, img2])
cv2.namedWindow("Manual matching")
cv2.resizeWindow("Manual Matching", 1000, 1000)
cv2.setMouseCallback("Manual matching", on_EVENT_LBUTTONDOWN)
#cv2.line(img, pt1, pt2, color[, thickness[, lineType[, shift]]])

cv2.imshow('Manual matching', imgs)
cv2.waitKey(0)