import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
img = cv.imread('../../img/sudoku.png')
blur = cv.blur(img,(5,5))
plt.subplot(321),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(322),plt.imshow(blur),plt.title('Blurred')
plt.xticks([]), plt.yticks([])
gblur = cv.GaussianBlur(img,(5,5),0)
plt.subplot(323),plt.imshow(blur),plt.title('Gaussian Blurred')
plt.xticks([]), plt.yticks([])
mblur = cv.medianBlur(img,5)
plt.subplot(324),plt.imshow(blur),plt.title('Median Blurred')
plt.xticks([]), plt.yticks([])
bfilter = cv.bilateralFilter(img,9,75,75)
plt.subplot(325),plt.imshow(bfilter),plt.title('BilateralFilter')
plt.xticks([]), plt.yticks([])
plt.show()