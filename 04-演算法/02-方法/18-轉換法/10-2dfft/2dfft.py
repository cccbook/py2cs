import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift

# 讀入影像
img = plt.imread('image.png')

# 將影像進行傅立葉變換
f = fft2(img)
fshift = fftshift(f)

# 繪製頻譜圖
magnitude_spectrum = 20*np.log(np.abs(fshift))

plt.figure()
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum')
plt.show()





