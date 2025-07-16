import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 設定隨機種子
np.random.seed(0)
torch.manual_seed(0)

# 生成二元分類的訓練數據
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

# 轉換為 PyTorch 張量
X = torch.FloatTensor(X)
y = torch.FloatTensor(y.reshape(-1, 1))

# 定義邏輯回歸模型
class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(2, 1)  # 2個特徵，1個輸出
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        return self.sigmoid(self.linear(x))

# 初始化模型、損失函數和優化器
model = LogisticRegression()
criterion = nn.BCELoss()  # Binary Cross Entropy Loss
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

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
        # 計算準確率
        predictions = (y_pred > 0.5).float()
        accuracy = (predictions == y).float().mean()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, Accuracy: {accuracy.item():.4f}")

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
x_min, x_max = X[:, 0].min().item() - 1, X[:, 0].max().item() + 1
y_min, y_max = X[:, 1].min().item() - 1, X[:, 1].max().item() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))
grid = np.c_[xx.ravel(), yy.ravel()]

# 預測網格點的類別
grid_torch = torch.FloatTensor(grid)
with torch.no_grad():
    Z = model(grid_torch).numpy()
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
print(f"Weights: {model.linear.weight.data.numpy()}")
print(f"Bias: {model.linear.bias.item():.4f}")

# 計算最終準確率
with torch.no_grad():
    final_predictions = (model(X) > 0.5).float()
    final_accuracy = (final_predictions == y).float().mean()
print(f"\n最終準確率: {final_accuracy.item():.4f}")