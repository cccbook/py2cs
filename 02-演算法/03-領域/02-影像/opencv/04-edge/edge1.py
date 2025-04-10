import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
img = cv.imread('../../img/footballman.png',0)
plt.subplot(221),plt.imshow(img)
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(223),plt.imshow(img,cmap = 'gray')
plt.title('Gray'), plt.xticks([]), plt.yticks([])
edges = cv.Canny(img,100,200)
plt.subplot(224),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()