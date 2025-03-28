import cv2
import matplotlib.pyplot as plt

filename = 'lena.png'
img = cv2.imread(filename)
plt.imshow(img, cmap = 'gray', interpolation = 'bicubic') # 
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()

'''
注意：以下的 cv2 功能測試失敗

print(type(img))
# print(type(img), img.shape, img.dtype)
# cv2.namedWindow('demo1')
cv2.imshow('demo1', img)
cv2.waitKey(0)

解決方式

https://stackoverflow.com/questions/14655969/opencv-error-the-function-is-not-implemented

Don't waste your time trying to resolve this issue, this was made clear by the makers themselves. Instead of cv2.imshow() use this:

img = cv2.imread('path_to_image')
plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()

'''