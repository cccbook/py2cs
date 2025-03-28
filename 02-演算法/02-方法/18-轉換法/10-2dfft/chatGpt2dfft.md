請產生一個可以將影像的 2d 傅立葉轉換頻譜用 matplotlib 繪圖的程式
下面是一個使用 Python 和 Matplotlib 繪製 2D 影像的傅立葉變換頻譜的範例程式碼:

Copy code
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
程式碼中讀入影像檔案並用 fft2 進行 2D 傅立葉變換，再將頻譜平移到中心使用 fftshift，最後使用 Matplotlib 繪製頻譜圖。
影像檔案名稱請自行替換為實際存在的檔案.