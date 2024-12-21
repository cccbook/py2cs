import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset, random_split

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

# 將數據包裝為 Dataset 和 DataLoader
dataset = TensorDataset(X, y)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# 定義邏輯回歸模型
class LogisticRegressionModel(pl.LightningModule):
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(2, 1)  # 2個特徵，1個輸出
        self.sigmoid = nn.Sigmoid()
        self.criterion = nn.BCELoss()

    def forward(self, x):
        return self.sigmoid(self.linear(x))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.criterion(y_pred, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.criterion(y_pred, y)
        predictions = (y_pred > 0.5).float()
        accuracy = (predictions == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_accuracy", accuracy, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)

# 初始化模型
model = LogisticRegressionModel()

# 訓練模型
trainer = pl.Trainer(max_epochs=100, log_every_n_steps=10)
trainer.fit(model, train_loader, val_loader)

# 繪製數據點和決策邊界
plt.figure(figsize=(10, 5))

# 創建網格來繪製決策邊界
x_min, x_max = X[:, 0].min().item() - 1, X[:, 0].max().item() + 1
y_min, y_max = X[:, 1].min().item() - 1, X[:, 1].max().item() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))
grid = np.c_[xx.ravel(), yy.ravel()]

# 預測網格點的類別
grid_torch = torch.FloatTensor(grid)
model.eval()
with torch.no_grad():
    Z = model(grid_torch).numpy()
Z = Z.reshape(xx.shape)

# 繪製決策邊界
plt.contour(xx, yy, Z, levels=[0.5], colors='k', linestyles='-')
plt.scatter(X1[:, 0], X1[:, 1], c='blue', label='Class 0')
plt.scatter(X2[:, 0], X2[:, 1], c='red', label='Class 1')
plt.title('Decision Boundary')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()
plt.show()
