import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from numpy import *
from pylab import *
from PIL import Image

def histeq(im,nbr_bins=256):
    """ 对一幅灰度图像进行直方图均衡化 """
    # 计算图像的直方图
    imhist,bins = histogram(im.flatten(),nbr_bins,normed=True)
    cdf = imhist.cumsum() # cumulative distribution function
    cdf = 255 * cdf / cdf[-1] # 归一化
    im2 = interp(im.flatten(),bins[:-1],cdf)
    return im2.reshape(im.shape), cdf

def adjust_color(scr, img0):
    img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)

    img = img0.copy()  # 用于之后做对比图
    scr = cv2.cvtColor(scr, cv2.COLOR_BGR2GRAY)
    mHist1 = []
    mNum1 = []
    inhist1 = []
    mHist2 = []
    mNum2 = []
    inhist2 = []

    # 对原图像进行均衡化
    for i in range(256):
        mHist1.append(0)

    row, col = img.shape  # 获取原图像像素点的宽度和高度
    print(img.size)
    for i in range(row):
        for j in range(col):
            mHist1[img[i, j]] = mHist1[img[i, j]] + 1  # 统计灰度值的个数
    mNum1.append(mHist1[0] / img.size)  # 每个intensity的占比

    for i in range(0, 255):
        mNum1.append(mNum1[i] + mHist1[i + 1] / img.size)

    for i in range(256):
        inhist1.append(round(255 * mNum1[i]))

    # 对目标图像进行均衡化
    for i in range(256):
        mHist2.append(0)

    rows, cols = scr.shape  # 获取目标图像像素点的宽度和高度
    for i in range(rows):
        for j in range(cols):
            mHist2[scr[i, j]] = mHist2[scr[i, j]] + 1  # 统计灰度值的个数
    mNum2.append(mHist2[0] / scr.size)

    for i in range(0, 255):
        mNum2.append(mNum2[i] + mHist2[i + 1] / scr.size)
    for i in range(256):
        inhist2.append(round(255 * mNum2[i]))

    # 进行规定化
    g = []  # 用于放入规定化后的图片像素
    for i in range(256):
        a = inhist1[i]
        flag = True
        for j in range(256):
            if inhist2[j] == a:
                g.append(j)
                flag = False
                break
        if flag == True:
            minp = 255
            for j in range(256):
                b = abs(inhist2[j] - a)
                if b < minp:
                    minp = b
                    jmin = j
            g.append(jmin)

    for i in range(row):
        for j in range(col):
            img[i, j] = g[img[i, j]]
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(5, 5))
    scr = clahe.apply(scr)
    img = clahe.apply(img)

    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

if __name__ == '__main__':
    img1_path = str('D:\Code\Rota_stitching\Export GEN Extracted HiMy 20200521\CMC 637_20170703_110246_R_Mac\CMC 637_20170703_110246_R_RMapEnSumRNFL_3Resized.png').replace('\\\\', '\\')
    img2_path = str('D:\Code\Rota_stitching\Export GEN Extracted HiMy 20200521\CMC 637_20170703_110757_R_Opt\CMC 637_20170703_110757_R_RMapEnSumRNFL_3Resized.png').replace('\\\\', '\\')

    scr = cv2.imread(img1_path)
    img0 = cv2.imread(img2_path)

    adjust_color(scr, img0)
