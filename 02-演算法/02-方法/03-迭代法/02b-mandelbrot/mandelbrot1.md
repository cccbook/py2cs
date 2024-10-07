### **Mandelbrot 集合的固定點迭代**
   Mandelbrot 集合是一個複數平面上的集合，其生成是基於固定點迭代。對於每個複數 \( c \)，進行以下迭代：
   \[
   z_{n+1} = z_n^2 + c
   \]
   如果 \( z_n \) 不會趨於無窮大，則 \( c \) 屬於 Mandelbrot 集合。這個迭代過程的固定點是那些使 \( z_n \) 保持有界的 \( c \) 值。

## ccc:請用 python 寫一個 Mandelbrot 集合的固定點迭代的程式

Mandelbrot 集合是透過複數的固定點迭代來生成的集合。Mandelbrot 集合的核心迭代公式為：

\[
z_{n+1} = z_n^2 + c
\]

其中，\( z_0 = 0 \)，\( c \) 是一個複數。如果迭代後 \( |z_n| \) 保持有界，那麼 \( c \) 屬於 Mandelbrot 集合。

以下是用 Python 來實現這個固定點迭代並生成 Mandelbrot 集合的程式：

### Python 程式：Mandelbrot 集合的固定點迭代

```python
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
```

### 說明：

1. **`mandelbrot` 函數**：這是核心的固定點迭代函數。它接收複數 \( c \) 和最大迭代次數，對每個複數進行固定點迭代，判斷該點是否屬於 Mandelbrot 集合。如果 \( z_n \) 的絕對值在迭代過程中大於 2，則認為不屬於集合，並返回當前迭代次數（用於著色）。如果迭代未超過最大迭代次數，則認為該點屬於 Mandelbrot 集合。

2. **`mandelbrot_set` 函數**：生成一個給定範圍內的複數平面，對每個複數進行迭代計算。它返回一個包含著色值的陣列，用於繪製 Mandelbrot 集合。

3. **`plot_mandelbrot` 函數**：使用 `matplotlib` 繪製 Mandelbrot 集合，並使用顏色來區分不同的迭代結果。超過最大迭代次數的點顯示為集合的一部分，其他點顯示不同的迭代次數。

### 程式結果：
這段程式碼會生成一個 800x800 像素的 Mandelbrot 集合圖像，並根據迭代次數用不同的顏色繪製。生成的圖像將展示出 Mandelbrot 集合的特徵性分形圖案。

你可以調整 `xmin`、`xmax`、`ymin`、`ymax` 的範圍來縮放圖像，或增加 `width` 和 `height` 來提高解析度。如果需要更詳細的分形結構，還可以增加 `max_iter` 的值來進行更深的迭代。