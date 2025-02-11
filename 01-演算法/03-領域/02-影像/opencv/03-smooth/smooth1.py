import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
img0 = cv.imread('../../img/ccc.png')
img = cv.imread('../../img/ccc.png')
# kernel = np.ones((5,5),np.float32)/25
# dst = cv.filter2D(img,-1,kernel)
dst = img
plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
#plt.subplot(122),plt.imshow(dst),plt.title('Averaging')
#plt.xticks([]), plt.yticks([])
plt.show()