import gd  # 假設已經有 gd.gradientDescendent 函式
import numpy as np

# 七段顯示器的真值表（輸入：7 位元，輸出：4 位元二進制）
seven_segment_truth_table = {
    0:  (1, 1, 1, 1, 1, 1, 0),
    1:  (0, 1, 1, 0, 0, 0, 0),
    2:  (1, 1, 0, 1, 1, 0, 1),
    3:  (1, 1, 1, 1, 0, 0, 1),
    4:  (0, 1, 1, 0, 0, 1, 1),
    5:  (1, 0, 1, 1, 0, 1, 1),
    6:  (1, 0, 1, 1, 1, 1, 1),
    7:  (1, 1, 1, 0, 0, 0, 0),
    8:  (1, 1, 1, 1, 1, 1, 1),
    9:  (1, 1, 1, 1, 0, 1, 1)
}

# 目標輸出 (數字的 4 位元二進位表示)
binary_outputs = {
    0: (0, 0, 0, 0),
    1: (0, 0, 0, 1),
    2: (0, 0, 1, 0),
    3: (0, 0, 1, 1),
    4: (0, 1, 0, 0),
    5: (0, 1, 0, 1),
    6: (0, 1, 1, 0),
    7: (0, 1, 1, 1),
    8: (1, 0, 0, 0),
    9: (1, 0, 0, 1)
}

# 建立訓練資料
input_vectors = np.array([seven_segment_truth_table[i] for i in range(10)])  # 10 x 7
target_outputs = np.array([binary_outputs[i] for i in range(10)])  # 10 x 4

# 權重矩陣 (7x4)，初始值為隨機數
weights = np.random.rand(7, 4)

# 誤差函數：均方誤差 (MSE)
def loss_function(w):
    # w = np.array(w).reshape(7, 4)  # 確保 w 是 7x4 矩陣
    predictions = input_vectors @ w  # 預測輸出
    return np.mean((predictions - target_outputs) ** 2)  # MSE

# 使用梯度下降法來調整權重
p = weights.flatten().tolist()  # 攤平成 1D 列表
gd.gradientDescendent(loss_function, p)

# 訓練後的權重
trained_weights = np.array(p).reshape(7, 4)

# 預測函數
def predict(segment_input):
    prediction = np.round(segment_input @ trained_weights).astype(int)  # 轉換為 0/1
    return prediction

# 測試預測
for num, segment in seven_segment_truth_table.items():
    binary_prediction = predict(np.array(segment))
    binary_str = "".join(map(str, binary_prediction))
    print(f"Input: {segment} -> Predicted Binary: {binary_str}")
