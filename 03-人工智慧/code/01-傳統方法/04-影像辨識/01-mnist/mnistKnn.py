# 載入需要的庫
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 載入 MNIST 數據集
digits = datasets.load_digits()

# 查看數據集的基本情況
print(f"數據集大小: {digits.data.shape}")
print(f"目標標籤: {np.unique(digits.target)}")

# 分割數據集為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.3, random_state=42)

# 特徵標準化（對特徵縮放進行預處理）
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 使用 K-NN 分類器進行訓練
knn = KNeighborsClassifier(n_neighbors=3)  # 設定使用 3 個鄰居
knn.fit(X_train, y_train)

# 在測試集上進行預測
y_pred = knn.predict(X_test)

# 計算並輸出準確率
accuracy = accuracy_score(y_test, y_pred)
print(f"K-NN 模型準確率: {accuracy * 100:.2f}%")

# 顯示一些預測結果
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
axes = axes.ravel()

for i in np.arange(10):
    axes[i].imshow(X_test[i].reshape(8, 8), cmap='gray')
    axes[i].set_title(f'predict: {y_pred[i]} label: {y_test[i]}')
    axes[i].axis('off')

plt.show()