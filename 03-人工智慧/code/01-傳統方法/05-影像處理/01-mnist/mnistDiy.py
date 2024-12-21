from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import random
# 載入 MNIST 數據集
digits = datasets.load_digits()

# 取一張圖像進行測試
i = random.randint(0, 1000)
image = digits.images[i]

# 將數字圖像轉換為灰階圖像（已經是灰階）
gray_image = image

# 二值化處理（將圖像轉換為黑白兩色）
threshold = 7  # 設定閾值
binary_image = np.where(gray_image > threshold, 255, 0)

# 顯示二值化圖像
plt.imshow(binary_image, cmap='gray')
plt.title('Binary Image')
plt.show()

# 這裡簡單地假設你已經有了數字模板，這可以是從 MNIST 或其他來源獲取的標準圖像
# 假設這是數字模板 (0-9)，這是簡單範例中的一個模板圖像
template = np.zeros((8, 8), dtype=np.uint8)  # 在實際應用中，這應該是數字模板

# 基本的模式匹配：與模板圖像進行簡單的比較
def match_template(image, template):
    """簡單的模板匹配，使用均方誤差來比較圖像與模板"""
    # 確保圖像和模板的大小一致
    if image.shape != template.shape:
        return float('inf')  # 大小不一致，返回無窮大

    # 計算均方誤差（MSE）
    mse = np.mean((image - template) ** 2)
    return mse

# 假設我們有一個模板列表（0-9的模板），這裡僅作示範
templates = [np.zeros((8, 8), dtype=np.uint8) for _ in range(10)]  # 用空白模板作示範

# 執行匹配，找出最匹配的模板
min_mse = float('inf')
best_match = -1
for i, tmpl in enumerate(templates):
    mse = match_template(binary_image, tmpl)
    if mse < min_mse:
        min_mse = mse
        best_match = i

print(f"預測結果：數字 {best_match}")
