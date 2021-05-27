import numpy as np
import cv2
import matplotlib.pyplot as plt
std_points = [[709.98846, 2543.5789], [608.76697, 2617.945], [691.5752, 2930.7295], [1290.5787, 2976.9832], [682.6171, 2902.77], [692.00867, 2949.3523], [1143.2653, 3188.7104], [820.73175, 2359.4258], [691.5387, 2424.2288], [774.2042, 2756.0942], [1318.3789, 2866.442], [2239.473, 3050.2793], [1797.4457, 2332.0408], [1868.6531, 1891.7283], [1718.4049, 2083.2622], [838.70557, 2368.8782], [700.7308, 2442.4282], [765.5579, 2755.745], [1281.4097, 2930.9387], [2055.3777, 2395.9683], [1954.2871, 2461.0542], [2027.6661, 2774.1113], [2682.208, 2839.0107], [3483.7332, 2405.606], [3916.9656, 2036.9923], [4156.871, 1922.0352], [3896.1394, 1848.4956]]
usr_points = [[236.71942, 257.64838], [202.82614, 241.98062], [161.00685, 271.9778], [163.6437, 393.37372], [287.6018, 222.34145], [208.00673, 254.97763], [202.81071, 367.23718], [290.26456, 275.8574], [275.8898, 256.3061], [209.33572, 283.7137], [205.42406, 365.94684], [359.41763, 392.09204], [301.97534, 328.09702], [284.16537, 349.38113], [290.46844, 308.12466], [295.47842, 274.56897], [296.83167, 257.64246], [213.21994, 249.77419], [179.31606, 386.8398], [286.2876, 256.30286], [287.62003, 236.75], [193.67725, 256.27625], [201.49197, 377.68253], [420.7656, 407.69983], [539.5531, 375.09195], [569.01013, 397.6194], [569.8221, 400.86697]]

b = np.ones(len(std_points))
print(len(std_points))
std_1 = np.c_[std_points,b.T]
use_1 = np.c_[usr_points,b.T]
#print(std_1)
#std_1_T = std_1.T
#print(std_1_T[:,1:2])
#F = [[ 1.04468998, -1.36687801, -3.22140929],[-9.07143333, 1.65223753, -4.21189346],[ 3.30064650, -4.19359424, 1.00000]]
M, mask = cv2.findHomography(np.float32(std_points), np.float32(usr_points), cv2.RANSAC)
print(M)
i = 0
error = 0
estimate_point = []
while i < len(std_points):
    estimate_point.append(np.dot(M, std_1[i].T))
    #single_e = np.abs(np.dot(np.dot(std_1[i:i+1], F), use_1[i:i+1].T))
    #error+=single_e
    i+=1
estimate_point = np.delete(estimate_point, -1, axis=1)
msex = ((np.float32(estimate_point) - np.float32(usr_points)) ** 2).mean(axis=0)
#msey = ((np.float32(estimate_point) - np.float32(usr_points)) ** 2).mean(axis=0)
print(np.float32(estimate_point))
#print(error)
print(std_1[26:27].T.shape)
#print (np.transpose(std_1,(1,0)))
print(use_1[26:27].shape)
#print(np.dot(use_1[26:27],std_1[26:27].T))
print(msex)
print(msex[0]*msex[1])
#print(msey)
#print(msey*msex)
print(5.0e+09)
list = [5.0e+09, 1.3e+08, 2.6e+10, 3.4e+09]
#plt.plot(list)
#plt.legend()
#plt.show()
i1 = cv2.imread('Bobath-1.jpg')
i2 = cv2.imread('Bobath-2.jpg')

cv2.imshow('1',i1)
cv2.imshow('2',i2)
cv2.waitKey()