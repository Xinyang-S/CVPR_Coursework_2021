import cv2
import numpy as np
width = 700
height = 700
length = 100
image = np.zeros((width,height),dtype = np.uint8)
print(image.shape[0],image.shape[1])

for j in range(height):
    for i in range(width):
        if((int)(i/length) + (int)(j/length))%2:
            image[i,j] = 255;
cv2.imwrite(r"D:\IC-Msc\Computer Vision\coursework1\cardbox1.jpg",image)
cv2.imshow("Cardbox",image)
cv2.waitKey(0)