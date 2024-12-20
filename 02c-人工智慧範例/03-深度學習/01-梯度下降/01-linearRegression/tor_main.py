import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 設定隨機種子
np.random.seed(0)
torch.manual_seed(0)

# 生成訓練數據
X = np.linspace(-10, 10, 100)
y = 2 * X + 1 + np.random.normal(0, 1.5, 100)  # y = 2x + 1 + noise

# 轉換為 PyTorch 張量並重塑
X = torch.FloatTensor(X.reshape(-1, 1))
y = torch.FloatTensor(y.reshape(-1, 1))

# 定義線性回歸模型
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)  # 輸入維度=1，輸出維度=1
    
    def forward(self, x):
        return self.linear(x)

# 初始化模型、損失函數和優化器
model = LinearRegression()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# 訓練模型
n_epochs = 1000
losses = []

for epoch in range(n_epochs):
    # 前向傳播
    y_pred = model(X)
    loss = criterion(y_pred, y)
    
    # 反向傳播和優化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # 儲存損失值
    losses.append(loss.item())
    
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# 獲取最終模型參數
w_final = model.linear.weight.item()
b_final = model.linear.bias.item()
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
X_np = X.numpy()
y_np = y.numpy()
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