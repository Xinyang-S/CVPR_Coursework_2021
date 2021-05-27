import cv2 as cv
from matplotlib import pyplot as plt
imgL = cv.imread(r'D:\IC-Msc\Computer Vision\Original Picture\FD\WithObject\DSCF0393.JPG',0)
imgR = cv.imread(r'D:\IC-Msc\Computer Vision\Original Picture\FD\WithObject\DSCF0404.JPG',0)
stereo = cv.StereoBM_create(numDisparities=16, blockSize=5)
disparity = stereo.compute(imgL,imgR)
plt.imshow(disparity,'gray')
plt.colorbar()
plt.show()

win_size = 2
min_disp = -4
max_disp = 9
num_disp = max_disp - min_disp  # Needs to be divisible by 16
stereo = cv.StereoSGBM_create(
    minDisparity=min_disp,
    numDisparities=num_disp,
    blockSize=5,
    uniquenessRatio=5,
    speckleWindowSize=5,
    speckleRange=5,
    disp12MaxDiff=2,
    P1=8 * 3 * win_size ** 2,
    P2=32 * 3 * win_size ** 2,
)
disparity_SGBM = stereo.compute(imgL, imgR)
plt.imshow(disparity_SGBM, "gray")
plt.colorbar()
plt.show()