import os

import cv2
import numpy as np
import matplotlib.pyplot as plt


def extract_vessel_from_oct(img):
    # Convert into gery scale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.medianBlur(img, 5)
    plt.imshow(img)
    plt.show()
    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    contrast_enhanced_green_fundus = clahe.apply(img)
    plt.imshow(contrast_enhanced_green_fundus)
    plt.show()
    # Morphology analysis: Three times closing and opening
    r1 = cv2.morphologyEx(contrast_enhanced_green_fundus, cv2.MORPH_OPEN,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    plt.imshow(r1)
    plt.show()
    R1 = cv2.morphologyEx(r1, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                          iterations=1)
    plt.imshow(R1)
    plt.show()
    r2 = cv2.morphologyEx(R1, cv2.MORPH_OPEN,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=1)
    plt.imshow(r2)
    plt.show()
    R2 = cv2.morphologyEx(r2, cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=1)
    plt.imshow(R2)
    plt.show()
    r3 = cv2.morphologyEx(R2, cv2.MORPH_OPEN,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)), iterations=1)
    R3 = cv2.morphologyEx(r3, cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)), iterations=1)
    # Image subtract
    f4 = cv2.subtract(R2, contrast_enhanced_green_fundus)
    # CLAHE
    f5 = clahe.apply(f4)
    # removing very small contours through area parameter noise removal
    ret, f6 = cv2.threshold(f5, 15, 255, cv2.THRESH_BINARY)
    mask = np.ones(f5.shape[:2], dtype="uint8") * 255
    _, contours, hierarchy = cv2.findContours(f6.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) <= 100:
            cv2.drawContours(mask, [cnt], -1, 0, -1)
    # bitwise_and and binarlization
    im = cv2.bitwise_and(f5, f5, mask=mask)
    ret, fin = cv2.threshold(im, 15, 255, cv2.THRESH_BINARY_INV)
    newfin = cv2.erode(fin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    return newfin


def extract_bv(image):

    b, green_fundus, r = cv2.split(image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3, 3))
    contrast_enhanced_green_fundus = clahe.apply(green_fundus)
    plt.imshow(contrast_enhanced_green_fundus)
    plt.show()
    # applying alternate sequential filtering (3 times closing opening)
    r1 = cv2.morphologyEx(contrast_enhanced_green_fundus, cv2.MORPH_OPEN,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    R1 = cv2.morphologyEx(r1, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    r2 = cv2.morphologyEx(R1, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=1)
    R2 = cv2.morphologyEx(r2, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=1)
    plt.imshow(R2)
    plt.show()
    r3 = cv2.morphologyEx(R2, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)), iterations=1)

    R3 = cv2.morphologyEx(r3, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)), iterations=1)
    f4 = cv2.subtract(R3, contrast_enhanced_green_fundus)
    f5 = clahe.apply(f4)
    plt.imshow(f5)
    plt.show()
    # removing very small contours through area parameter noise removal
    ret, f6 = cv2.threshold(f5, 15, 255, cv2.THRESH_BINARY)
    mask = np.ones(f5.shape[:2], dtype="uint8") * 255
    contours, hierarchy = cv2.findContours(f6.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) <= 40:
            cv2.drawContours(mask, [cnt], -1, 0, -1)
    im = cv2.bitwise_and(f5, f5, mask=mask)
    ret, fin = cv2.threshold(im, 15, 255, cv2.THRESH_BINARY_INV)
    newfin = cv2.erode(fin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)

    # removing blobs of unwanted bigger chunks taking in consideration they are not straight lines like blood
    # vessels and also in an interval of area
    fundus_eroded = cv2.bitwise_not(newfin)
    xmask = np.ones(fundus_eroded.shape[:2], dtype="uint8") * 255
    xcontours, xhierarchy = cv2.findContours(fundus_eroded.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in xcontours:
        shape = "unidentified"
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, False)
        if len(approx) > 4 and cv2.contourArea(cnt) <= 3000 and cv2.contourArea(cnt) >= 10:
            shape = "circle"
        else:
            shape = "veins"
        if (shape == "circle"):
            cv2.drawContours(xmask, [cnt], -1, 0, -1)
    plt.imshow(xmask)
    plt.show()
    finimage = cv2.bitwise_and(fundus_eroded, fundus_eroded, mask=xmask)
    blood_vessels = cv2.bitwise_not(finimage)
    return xmask

def boundary_detection(img):
    image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 构造一个3×3的结构元素
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilate = cv2.dilate(image, element)
    erode = cv2.erode(image, element)

    # 将两幅图像相减获得边，第一个参数是膨胀后的图像，第二个参数是腐蚀后的图像
    result = cv2.absdiff(dilate, erode);

    # 上面得到的结果是灰度图，将其二值化以便更清楚的观察结果
    retval, result = cv2.threshold(result, 40, 255, cv2.THRESH_BINARY);
    # 反色，即对二值图每个像素取反
    result = cv2.bitwise_not(result);
    # 显示图像
    cv2.imshow("result", result);
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return result

def gradient_filter(img):
    import numpy as np
    import cv2 as cv
    from matplotlib import pyplot as plt
    laplacian = cv.Laplacian(img, cv.CV_64F)
    sobelx = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=11)
    sobely = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=11)
    plt.subplot(2, 2, 1), plt.imshow(img, cmap='gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 2), plt.imshow(laplacian, cmap='gray')
    plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 3), plt.imshow(sobelx, cmap='gray')
    plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 4), plt.imshow(sobely, cmap='gray')
    plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
    plt.show()

if __name__ == '__main__':
    folder = './Export GEN Extracted HiMy 20200521/'
    for file_id in os.listdir(folder):
        if 'EnfaceAdjusted.png' in os.listdir(folder + file_id):
            print(True)
            img = cv2.imread(folder + file_id + '/' + 'EnfaceAdjusted.png')
            print(np.shape(img))
            extracted_img = extract_vessel_from_oct(img)
            extracted_img = cv2.merge((extracted_img, extracted_img, extracted_img))
            displayed_img = np.concatenate((img, extracted_img), axis=1)
            plt.imshow(displayed_img)
            plt.show()