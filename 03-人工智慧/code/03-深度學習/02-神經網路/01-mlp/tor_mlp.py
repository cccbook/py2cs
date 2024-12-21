import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

def generate_data(n_samples=100, seed=0):
    """生成二元分類的訓練數據"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    
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
    
    return X, y, X1, X2

class SimpleMLP(nn.Module):
    """簡單的多層感知器模型"""
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2, 16)
        self.layer2 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        return self.sigmoid(self.layer2(x))

def accuracy(model, X, y):
    """計算準確率"""
    with torch.no_grad():
        pred = model(X)
        pred = (pred > 0.5).float()
        return (pred == y).float().mean()

def plot_results(model, X, y, X1, X2, losses):
    """繪製訓練結果"""
    plt.figure(figsize=(15, 5))

    # 繪製訓練損失
    plt.subplot(1, 3, 1)
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Binary Cross Entropy Loss')

    # 繪製決策邊界
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

def main():
    # 設定超參數
    num_epochs = 1000
    learning_rate = 0.01
    
    # 生成數據
    X, y, X1, X2 = generate_data()
    
    # 創建模型
    model = SimpleMLP()
    
    # 創建損失函數和優化器
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    # 訓練模型
    losses = []
    for epoch in range(num_epochs):
        # 前向傳播
        outputs = model(X)
        loss = criterion(outputs, y)
        
        # 反向傳播和優化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 記錄損失
        losses.append(loss.item())
        
        if (epoch + 1) % 100 == 0:
            acc = accuracy(model, X, y)
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, Accuracy: {acc.item():.4f}")
    
    # 繪製結果
    plot_results(model, X, y, X1, X2, losses)
    
    # 輸出最終準確率
    final_accuracy = accuracy(model, X, y)
    print(f"\n最終準確率: {final_accuracy.item():.4f}")

if __name__ == "__main__":
    main()