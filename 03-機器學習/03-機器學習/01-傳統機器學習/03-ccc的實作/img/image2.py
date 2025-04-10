# 參考 https://matplotlib.org/3.1.1/tutorials/introductory/images.html#sphx-glr-tutorials-introductory-images-py
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img = mpimg.imread('ccc.jpg')
imgplot = plt.imshow(img)
plt.show()

lum_img = img[:, :, 0]
imgplot = plt.imshow(lum_img)
plt.show()

plt.imshow(lum_img, cmap="hot")
plt.show()
