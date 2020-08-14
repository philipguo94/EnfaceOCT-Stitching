import os

from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import matplotlib.pyplot as plt


images = []
# loop over the image paths, load each one, and add them to our
# images to stich list
for imagePath in os.listdir('images'):
	image = cv2.imread('./images/' + imagePath)
	images.append(image)
print(images)
# initialize OpenCV's image sticher object and then perform the image
# stitching
print("[INFO] stitching images...")
stitcher = cv2.createStitcher() if imutils.is_cv3() else cv2.Stitcher_create()
(status, stitched) = stitcher.stitch(images)
print(stitched)
plt.imshow(stitched)
plt.show()
cv2.imshow('stitched', stitched)
cv2.waitKey(0)
cv2.destroyAllWindows()