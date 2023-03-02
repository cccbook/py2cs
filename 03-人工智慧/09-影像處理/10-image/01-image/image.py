import numpy as np
import cv2

# 讀取圖檔
img = cv2.imread('image.jpg')

# 顯示圖片
cv2.imshow('My Image', img)

# 寫入圖片到檔案 
cv2.imwrite('image2.jpg', img)

# 按下任意鍵則關閉所有視窗
cv2.waitKey(0)
cv2.destroyAllWindows()

