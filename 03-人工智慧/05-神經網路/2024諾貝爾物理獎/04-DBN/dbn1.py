import numpy as np
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# 1. 加載 MNIST 資料集
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 將資料轉換為二進制 (0 或 1)
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)
X_train = (X_train > 127).astype(np.float32)
X_test = (X_test > 127).astype(np.float32)

# 樣本縮放至 [0, 1] 範圍
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 2. 設置 RBM 的數量和參數
rbm1 = BernoulliRBM(n_components=256, learning_rate=0.01, n_iter=10, random_state=0)
rbm2 = BernoulliRBM(n_components=128, learning_rate=0.01, n_iter=10, random_state=0)
rbm3 = BernoulliRBM(n_components=64, learning_rate=0.01, n_iter=10, random_state=0)

# 3. 設置最終的 Logistic Regression 用於監督學習微調
logistic = LogisticRegression(max_iter=1000, solver='lbfgs', random_state=0)

# 4. 使用管道來組合 RBM 和 Logistic Regression (形成 DBN)
dbn = Pipeline(steps=[('rbm1', rbm1), ('rbm2', rbm2), ('rbm3', rbm3), ('logistic', logistic)])

# 5. 訓練 DBN 模型
dbn.fit(X_train, y_train)

# 6. 測試模型並打印準確度
accuracy = dbn.score(X_test, y_test)
print(f"DBN 測試準確度: {accuracy * 100:.2f}%")

# 7. 測試用例：將部分測試集圖片進行預測
sample_images = X_test[:5]
sample_labels = y_test[:5]

predicted_labels = dbn.predict(sample_images)

# 顯示測試圖片及其預測結果
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(sample_images[i].reshape(28, 28), cmap='gray')
    plt.title(f"Label: {sample_labels[i]}\nPredicted: {predicted_labels[i]}")
    plt.axis('off')
plt.show()
