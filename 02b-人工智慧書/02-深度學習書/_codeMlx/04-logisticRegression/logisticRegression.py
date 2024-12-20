import mlx.core as mx
import mlx.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 生成二元分類的訓練數據
np.random.seed(0)
n_samples = 100

# 生成第一個類別的數據（圍繞 [-2, -2] 的點）
X1 = np.random.randn(n_samples // 2, 2) + np.array([-2, -2])
y1 = np.zeros(n_samples // 2)

# 生成第二個類別的數據（圍繞 [2, 2] 的點）
X2 = np.random.randn(n_samples // 2, 2) + np.array([2, 2])
y2 = np.ones(n_samples // 2)

# 合併數據
X = np.vstack([X1, X2])
y = np.hstack([y1, y2])

# 轉換為 MLX 張量
X = mx.array(X, dtype=mx.float32)
y = mx.array(y.reshape(-1, 1), dtype=mx.float32)

# 定義模型參數
weights = mx.zeros((2, 1), dtype=mx.float32)  # 2個特徵，1個輸出
bias = mx.zeros((1,), dtype=mx.float32)

def sigmoid(x):
    return 1 / (1 + mx.exp(-x))

def logistic_regression(x, w, b):
    return sigmoid(mx.matmul(x, w) + b)

def binary_cross_entropy(y_pred, y_true):
    epsilon = 1e-15  # 防止取對數時出現0
    y_pred = mx.clip(y_pred, epsilon, 1 - epsilon)
    return -mx.mean(y_true * mx.log(y_pred) + (1 - y_true) * mx.log(1 - y_pred))

# 定義訓練步驟
def train_step(x, y_true, w, b):
    def loss_fn(w, b):
        y_pred = logistic_regression(x, w, b)
        return binary_cross_entropy(y_pred, y_true)
    
    loss, grads = mx.value_and_grad(loss_fn, argnums=[0, 1])(w, b)
    return loss, grads

# 訓練模型
n_epochs = 1000
learning_rate = 0.1
losses = []

for epoch in range(n_epochs):
    # 計算損失和梯度
    loss, (w_grad, b_grad) = train_step(X, y, weights, bias)
    
    # 更新參數
    weights = weights - learning_rate * w_grad
    bias = bias - learning_rate * b_grad
    
    # 儲存損失值
    losses.append(float(loss))
    
    if (epoch + 1) % 100 == 0:
        # 計算準確率
        y_pred = logistic_regression(X, weights, bias)
        predictions = (y_pred > 0.5).astype(mx.float32)
        accuracy = mx.mean((predictions == y).astype(mx.float32))
        print(f"Epoch {epoch+1}, Loss: {float(loss):.4f}, Accuracy: {float(accuracy):.4f}")

# 繪製結果
plt.figure(figsize=(15, 5))

# 繪製訓練損失
plt.subplot(1, 3, 1)
plt.plot(losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Binary Cross Entropy Loss')

# 繪製數據點和決策邊界
plt.subplot(1, 3, 2)
plt.scatter(X1[:, 0], X1[:, 1], c='blue', label='Class 0')
plt.scatter(X2[:, 0], X2[:, 1], c='red', label='Class 1')

# 創建網格來繪製決策邊界
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))
grid = np.c_[xx.ravel(), yy.ravel()]

# 預測網格點的類別
grid_mx = mx.array(grid, dtype=mx.float32)
Z = logistic_regression(grid_mx, weights, bias)
Z = Z.reshape(xx.shape)

# 繪製決策邊界
plt.contour(xx, yy, Z, levels=[0.5], colors='k', linestyles='-')
plt.title('Decision Boundary')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()

# 繪製預測概率
plt.subplot(1, 3, 3)
plt.contourf(xx, yy, Z, levels=np.linspace(0, 1, 11), cmap='RdBu')
plt.colorbar(label='Probability of Class 1')
plt.scatter(X1[:, 0], X1[:, 1], c='blue', label='Class 0')
plt.scatter(X2[:, 0], X2[:, 1], c='red', label='Class 1')
plt.title('Prediction Probabilities')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()

plt.tight_layout()
plt.show()

# 打印最終模型參數
print("\n最終模型參數:")
print(f"Weights: {weights.tolist()}")
print(f"Bias: {float(bias)}")

# 計算最終準確率
final_predictions = (logistic_regression(X, weights, bias) > 0.5).astype(mx.float32)
final_accuracy = mx.mean((final_predictions == y).astype(mx.float32))
print(f"\n最終準確率: {float(final_accuracy):.4f}")
