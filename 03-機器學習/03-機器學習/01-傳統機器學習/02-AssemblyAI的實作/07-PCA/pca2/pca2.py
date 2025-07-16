import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 步驟 1：建立數據集
data = {
    'Height': [170, 160, 180],
    'Weight': [65, 50, 70]
}
df = pd.DataFrame(data)

# 步驟 2：數據標準化
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# 步驟 3：計算協方差矩陣
covariance_matrix = np.cov(scaled_data, rowvar=False)

# 步驟 4：特徵值分解
eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

# 步驟 5：選擇主成分
# 根據特徵值排序，選擇前 k 個主成分（這裡 k=2）
sorted_indices = np.argsort(eigenvalues)[::-1]
top_indices = sorted_indices[:2]
top_eigenvectors = eigenvectors[:, top_indices]

# 步驟 6：數據變換
pca_result = scaled_data.dot(top_eigenvectors)

# 輸出結果
print("原始數據標準化後：")
print(scaled_data)
print("\n協方差矩陣：")
print(covariance_matrix)
print("\n特徵值：")
print(eigenvalues)
print("\n選擇的主成分：")
print(top_eigenvectors)
print("\n降維後的數據：")
print(pca_result)
