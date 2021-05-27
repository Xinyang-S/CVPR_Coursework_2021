# coding:utf-8

import cv2

# 按照灰度图像读入两张图片
img1 = cv2.imread("D:/IC-Msc/Computer Vision/Original Picture/FD/WithObject/DSCF0398.jpg", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("D:/IC-Msc/Computer Vision/Original Picture/FG/WithObject/DSCF0399.jpg", cv2.IMREAD_GRAYSCALE)

# 获取特征提取器对象
orb = cv2.ORB_create()
# 检测关键点和特征描述
keypoint1, desc1 = orb.detectAndCompute(img1, None)
keypoint2, desc2 = orb.detectAndCompute(img2, None)
"""
keypoint 是关键点的列表
desc 检测到的特征的局部图的列表
"""
# 获得knn检测器
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.knnMatch(desc1, desc2, k=1)

"""
knn 匹配可以返回k个最佳的匹配项
bf返回所有的匹配项
"""
# 画出匹配结果
img3 = cv2.drawMatchesKnn(img1, keypoint1, img2, keypoint2, matches, img2, flags=2)
cv2.namedWindow("K-nearest",0)
cv2.imshow("K-nearest", img3)
cv2.waitKey()
cv2.destroyAllWindows()