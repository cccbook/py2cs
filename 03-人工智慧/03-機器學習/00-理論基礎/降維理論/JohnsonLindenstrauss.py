import numpy as np

def random_projection(X, k):
    """
    將高維數據矩陣 X 降維到 k 維
    :param X: n x d 的數據矩陣 (n 個樣本, 每個樣本 d 維)
    :param k: 目標降維的維度
    :return: n x k 的投影後的數據矩陣
    """
    n, d = X.shape
    # 生成隨機的 k x d 投影矩陣 R，每個元素服從標準正態分佈
    R = np.random.randn(k, d)
    # 投影數據並縮放
    Y = (1 / np.sqrt(k)) * np.dot(X, R.T)
    return Y

# 假設我們有 n = 100 個樣本，每個樣本在 d = 1000 維空間中
X = np.random.randn(100, 1000)

# 將其投影到 k = 50 維空間
Y = random_projection(X, k=50)

print("原始數據維度:", X.shape)
print("投影後數據維度:", Y.shape)
