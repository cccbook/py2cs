import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
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


class SimpleMLP(pl.LightningModule):
    """簡單的多層感知器模型"""
    def __init__(self, learning_rate=0.01):
        super().__init__()
        self.layer1 = nn.Linear(2, 16)
        self.layer2 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.criterion = nn.BCELoss()
        self.learning_rate = learning_rate

    def forward(self, x):
        x = self.relu(self.layer1(x))
        return self.sigmoid(self.layer2(x))

    def training_step(self, batch, batch_idx):
        X, y = batch
        y_pred = self(X)
        loss = self.criterion(y_pred, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.learning_rate)


def plot_results(model, X, y, X1, X2):
    """繪製訓練結果"""
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)

    # 繪製數據點
    plt.scatter(X1[:, 0], X1[:, 1], c="blue", label="Class 0")
    plt.scatter(X2[:, 0], X2[:, 1], c="red", label="Class 1")

    # 創建網格來繪製決策邊界
    x_min, x_max = X[:, 0].min().item() - 1, X[:, 0].max().item() + 1
    y_min, y_max = X[:, 1].min().item() - 1, X[:, 1].max().item() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid = np.c_[xx.ravel(), yy.ravel()]

    # 預測網格點的類別
    grid_torch = torch.FloatTensor(grid)
    with torch.no_grad():
        Z = model(grid_torch).numpy()
    Z = Z.reshape(xx.shape)

    # 繪製決策邊界
    plt.contour(xx, yy, Z, levels=[0.5], colors="k", linestyles="-")
    plt.title("Decision Boundary")
    plt.legend()
    plt.tight_layout()

    plt.subplot(1, 2, 2)
    plt.contourf(xx, yy, Z, levels=np.linspace(0, 1, 11), cmap="RdBu")
    plt.colorbar(label="Probability of Class 1")
    plt.scatter(X1[:, 0], X1[:, 1], c="blue", label="Class 0")
    plt.scatter(X2[:, 0], X2[:, 1], c="red", label="Class 1")
    plt.title("Prediction Probabilities")
    plt.tight_layout()
    plt.show()


def main():
    # 生成數據
    X, y, X1, X2 = generate_data()
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 訓練模型
    model = SimpleMLP()
    trainer = pl.Trainer(max_epochs=1000, log_every_n_steps=10)
    trainer.fit(model, dataloader)

    # 繪製結果
    plot_results(model, X, y, X1, X2)


if __name__ == "__main__":
    main()
