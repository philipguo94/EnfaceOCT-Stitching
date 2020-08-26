import os

import cv2
from psd_tools import PSDImage
import matplotlib.pyplot as plt
import PIL
import numpy as np


psd_path = 'D:\\Code\\Rota_stitching\\images\\calvin\\'
for filename in os.listdir(psd_path):
    if '.psd' in filename:
        psd = PSDImage.load(psd_path + filename)
        merged_image = psd.as_PIL()
        merged_image.save('./dataset/mask/' + filename[:-4]+'.png')
        img = cv2.imread(psd_path + filename[:-4]+'.jpg')
        cv2.imwrite('./dataset/img/' + filename[:-4]+'.png', img)

        img = cv2.imread('./dataset/mask/' + filename[:-4]+'.png')
        mask_vessel = np.array(img[:, :, 0]==0, dtype=np.uint8)
        contours, hierarchy = cv2.findContours(mask_vessel, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) <= 50:
                cv2.drawContours(mask_vessel, [cnt], -1, 0, -1)
        mask_vessel = np.array(mask_vessel*255, dtype=np.uint8)
        cv2.imwrite('./dataset/mask/' + filename[:-4]+'.png', mask_vessel)

        #img = psd.layers[0].as_PIL()
        #img = np.array(img)
        #cv2.imwrite('./dataset/mask/' + filename[:-4]+'.png', 255 - img[:,:,0])
        #img = cv2.imread(psd_path + filename[:-4]+'.jpg')
        #cv2.imwrite('./dataset/img/' + filename[:-4]+'.png', img)
