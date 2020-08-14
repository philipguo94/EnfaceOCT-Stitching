import os

import cv2
from psd_tools import PSDImage
import matplotlib.pyplot as plt
import PIL
import numpy as np


psd_path = 'D:\\Code\\Rota_stitching\\images\\'
for filename in os.listdir(psd_path):
    if '.psd' in filename:
        psd = PSDImage.load(psd_path + filename)
        img = psd.layers[0].as_PIL()
        img = np.array(img)
        cv2.imwrite('./dataset/mask/' + filename[:-4]+'.png', 255 - img[:,:,0])
        img = cv2.imread(psd_path + filename[:-4]+'.jpg')
        cv2.imwrite('./dataset/img/' + filename[:-4]+'.png', img)
