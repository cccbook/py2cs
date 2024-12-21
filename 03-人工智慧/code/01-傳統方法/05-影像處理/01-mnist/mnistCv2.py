import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

# 載入 MNIST 數據集
digits = datasets.load_digits()

# 取一個圖像進行測試
image = digits.images[0]

# 將數字圖像轉換為灰階圖像
gray_image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2GRAY)

# 二值化處理
_, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY_INV)

# 顯示二值化圖像
plt.imshow(binary_image, cmap='gray')
plt.title('Binary Image')
plt.show()

# 假設這是數字模板 (0-9)，這是簡單範例中的一個模板圖像
template = np.zeros((8, 8), dtype=np.uint8)  # 在實際應用中，這應該是數字模板

# 使用模板匹配
result = cv2.matchTemplate(binary_image, template, cv2.TM_CCOEFF_NORMED)

# 顯示匹配結果
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

# 在原圖上標記匹配區域
top_left = max_loc
bottom_right = (top_left[0] + template.shape[1], top_left[1] + template.shape[0])

# 畫矩形
cv2.rectangle(image, top_left, bottom_right, 255, 2)

# 顯示結果
plt.imshow(image, cmap='gray')
plt.title('Matched Image')
plt.show()
