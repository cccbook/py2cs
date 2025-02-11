import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import matplotlib.pyplot as plt

def generate_data(n_samples=100, seed=0):
    """生成二元分類的訓練數據"""
    np.random.seed(seed)
    
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
    
    return X, y, X1, X2

class SimpleMLP(nn.Module):
    """簡單的多層感知器模型"""
    def __init__(self):
        super().__init__()
        # 只使用兩個隱藏層
        self.layer1 = nn.Linear(2, 16)
        self.layer2 = nn.Linear(16, 1)

    def __call__(self, x):
        x = nn.relu(self.layer1(x))
        return nn.sigmoid(self.layer2(x))

def loss_fn(model, X, y):
    """損失函數"""
    pred = model(X)
    epsilon = 1e-15
    pred = mx.clip(pred, epsilon, 1 - epsilon)
    return -mx.mean(y * mx.log(pred) + (1 - y) * mx.log(1 - pred))

def accuracy(model, X, y):
    """計算準確率"""
    pred = model(X)
    pred = (pred > 0.5).astype(mx.float32)
    return mx.mean((pred == y).astype(mx.float32))

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
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    grid = np.c_[xx.ravel(), yy.ravel()]

    # 預測網格點的類別
    grid_mx = mx.array(grid, dtype=mx.float32)
    Z = model(grid_mx)
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
    
    # 創建優化器
    optimizer = optim.SGD(learning_rate=learning_rate)
    
    # 獲取損失和梯度的函數
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    
    # 訓練模型
    losses = []
    for epoch in range(num_epochs):
        # 計算損失和梯度
        loss, grads = loss_and_grad_fn(model, X, y)
        
        # 更新模型參數
        optimizer.update(model, grads)
        
        # 記錄損失
        losses.append(float(loss))
        
        if (epoch + 1) % 100 == 0:
            acc = accuracy(model, X, y)
            print(f"Epoch {epoch+1}, Loss: {float(loss):.4f}, Accuracy: {float(acc):.4f}")
    
    # 繪製結果
    plot_results(model, X, y, X1, X2, losses)
    
    # 輸出最終準確率
    final_accuracy = accuracy(model, X, y)
    print(f"\n最終準確率: {float(final_accuracy):.4f}")

if __name__ == "__main__":
    main()
