import numpy as np
import cv2
from matplotlib import pyplot as plt
MIN_MATCH_COUNT = 10
#原图
img_homo_rgb= cv2.imread('images/FP004_20151012_124452_R_Mac.jpg')
img_homo_rgb = cv2.medianBlur(img_homo_rgb, 5)
img_homo_rgb = img_homo_rgb[:125,125:]
img_homo_train_rgb = cv2.imread('images/FP004_20151012_124552_R_Opt.jpg')
img_homo_train_rgb = cv2.medianBlur(img_homo_train_rgb, 5)
img_homo_train_rgb = img_homo_train_rgb[:125, :125]

#灰度图
img_homo = cv2.imread('images/FP004_20151012_124452_R_Mac.jpg',0)
img_homo = cv2.medianBlur(img_homo, 5)
img_homo = img_homo[:125,125:]
img_homo_train = cv2.imread('images/FP004_20151012_124552_R_Opt.jpg',0)

img_homo_train = cv2.medianBlur(img_homo_train, 5)
img_homo_train = img_homo_train[:125, :125]


sift= cv2.ORB_create()

#SIFT特征点和特征描述提取
kp1, des1 = sift.detectAndCompute(img_homo,None)
kp2, des2 = sift.detectAndCompute(img_homo_train,None)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 100)
#Brute Force匹配和FLANN匹配是opencv二维特征点匹配常见的两种办法，分别对应BFMatcher和FlannBasedMatcher
flannbased = cv2.DescriptorMatcher_create(cv2.NORM_HAMMING)
matches = flannbased.knnMatch(des1,des2,k=2)

#使用KNN算法找到最近邻的两个数据点，
#如果最接近和次接近的比值大于一个既定的值，
#那么我们保留这个最接近的值，认为它和其匹配的点为good match
topmatch = []
for m,n in matches:
    if m.distance/n.distance < 1:
        topmatch.append(m)

if len(topmatch)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in topmatch ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in topmatch ]).reshape(-1,1,2)

    #其中M为求得的单应性矩阵矩阵
    #mask则返回一个列表来表征匹配成功的特征点。
    #src_pts,dst_pts为关键点
    #cv2.RANSAC, ransacReprojThreshold=5.0 这两个参数与RANSAC有关
    #RANSAC：随机抽样一致算法
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()

    h,w = img_homo.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    #透视变换(Perspective Transformation)是将成像投影到一个新的视平面(ViewingPlane)也称作投影映射
    #使用变换矩阵对原图的四个点获得在目标图像上的坐标
    dst = cv2.perspectiveTransform(pts,M)
    img_homo_train = cv2.polylines(img_homo_train,[np.int32(dst)],True,255,3, cv2.LINE_AA)
else:
    matchesMask = None

#特征点匹配连线
draw_params = dict(matchColor = (0,255,0),singlePointColor = None,matchesMask = matchesMask,flags = 2)
img_out = cv2.drawMatches(img_homo_rgb,kp1,img_homo_train_rgb,kp2,topmatch,None,**draw_params)
plt.imshow(img_out, 'Accent')
plt.show()