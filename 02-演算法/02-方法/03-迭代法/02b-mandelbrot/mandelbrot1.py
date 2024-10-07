import numpy as np
import matplotlib.pyplot as plt

# 定義 Mandelbrot 集合的參數
def mandelbrot(c, max_iter):
    z = 0
    for n in range(max_iter):
        if abs(z) > 2:
            return n  # 超過界限，返回迭代次數
        z = z**2 + c
    return max_iter  # 若未超過界限，則認為屬於集合

# 繪製 Mandelbrot 集合
def mandelbrot_set(xmin, xmax, ymin, ymax, width, height, max_iter):
    # 創建畫布，並根據給定的範圍生成複數平面
    r1 = np.linspace(xmin, xmax, width)
    r2 = np.linspace(ymin, ymax, height)
    mandelbrot_image = np.zeros((height, width))
    
    # 對每個點進行固定點迭代
    for i in range(width):
        for j in range(height):
            c = r1[i] + r2[j] * 1j  # 將座標轉換為複數
            mandelbrot_image[j, i] = mandelbrot(c, max_iter)
    
    return mandelbrot_image

# 繪製結果
def plot_mandelbrot(mandelbrot_image):
    plt.imshow(mandelbrot_image, cmap='inferno', extent=[xmin, xmax, ymin, ymax])
    plt.colorbar()
    plt.title('Mandelbrot Set')
    plt.show()

# 設定 Mandelbrot 集合的參數範圍和解析度
xmin, xmax, ymin, ymax = -2.0, 1.0, -1.5, 1.5
width, height = 800, 800
max_iter = 256

# 計算 Mandelbrot 集合
mandelbrot_image = mandelbrot_set(xmin, xmax, ymin, ymax, width, height, max_iter)

# 繪製 Mandelbrot 集合
plot_mandelbrot(mandelbrot_image)
