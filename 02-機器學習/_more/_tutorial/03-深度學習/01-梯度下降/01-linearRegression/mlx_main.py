import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import matplotlib.pyplot as plt

# 生成訓練數據
np.random.seed(0)
X = np.linspace(-10, 10, 100)
y = 2 * X + 1 + np.random.normal(0, 1.5, 100)  # y = 2x + 1 + noise

# 轉換為 MLX 張量並重塑
X = mx.array(X.reshape(-1, 1), dtype=mx.float32)
y = mx.array(y.reshape(-1, 1), dtype=mx.float32)

# 定義模型參數
weight = mx.array([[0.0]], dtype=mx.float32)  # 初始化權重
bias = mx.array([0.0], dtype=mx.float32)      # 初始化偏差

def linear_regression(x, w, b):
    return mx.matmul(x, w) + b

def mse_loss(y_pred, y_true):
    return mx.mean((y_pred - y_true) ** 2)

# 定義訓練步驟
def train_step(x, y_true, w, b):
    def loss_fn(w, b):
        y_pred = linear_regression(x, w, b)
        return mse_loss(y_pred, y_true)
    
    loss, grads = mx.value_and_grad(loss_fn, argnums=[0, 1])(w, b)
    return loss, grads

# 訓練模型
n_epochs = 1000
learning_rate = 0.001
losses = []

for epoch in range(n_epochs):
    # 計算損失和梯度
    loss, (w_grad, b_grad) = train_step(X, y, weight, bias)
    
    # 更新參數
    weight = weight - learning_rate * w_grad
    bias = bias - learning_rate * b_grad
    
    # 儲存損失值
    losses.append(float(loss))
    
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}, Loss: {float(loss):.4f}")

# 打印最終模型參數
w_final = float(weight[0,0])
b_final = float(bias[0])
print(f"\n最終模型: y = {w_final:.4f}x + {b_final:.4f}")

# 繪製結果
plt.figure(figsize=(10, 5))

# 繪製訓練過程
plt.subplot(1, 2, 1)
plt.plot(losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')

# 繪製擬合結果
plt.subplot(1, 2, 2)
X_np = X.tolist()
y_np = y.tolist()
plt.scatter(X_np, y_np, label='Data')
X_line = np.linspace(-10, 10, 100)
y_line = w_final * X_line + b_final
plt.plot(X_line, y_line, 'r', label=f'y = {w_final:.2f}x + {b_final:.2f}')
plt.title('Linear Regression Fit')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()

plt.tight_layout()
plt.show()