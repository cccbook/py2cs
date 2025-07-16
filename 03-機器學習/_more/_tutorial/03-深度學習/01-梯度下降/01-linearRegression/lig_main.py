import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

# 設定隨機種子
np.random.seed(0)
torch.manual_seed(0)

# 生成訓練數據
X = np.linspace(-10, 10, 100)
y = 2 * X + 1 + np.random.normal(0, 1.5, 100)  # y = 2x + 1 + noise

# 自定義 Dataset
class LinearDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X.reshape(-1, 1))
        self.y = torch.FloatTensor(y.reshape(-1, 1))
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

dataset = LinearDataset(X, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 定義 LightningModule
class LinearRegressionModule(pl.LightningModule):
    def __init__(self):
        super(LinearRegressionModule, self).__init__()
        self.linear = nn.Linear(1, 1)
        self.criterion = nn.MSELoss()
    
    def forward(self, x):
        return self.linear(x)
    
    def training_step(self, batch, batch_idx):
        X, y = batch
        y_pred = self.linear(X)
        loss = self.criterion(y_pred, y)
        self.log("train_loss", loss)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.001)

# 初始化模型
model = LinearRegressionModule()

# 定義 Trainer 並訓練
trainer = pl.Trainer(max_epochs=1000, log_every_n_steps=100, enable_checkpointing=False)
trainer.fit(model, dataloader)

# 獲取最終模型參數
w_final = model.linear.weight.item()
b_final = model.linear.bias.item()
print(f"\n最終模型: y = {w_final:.4f}x + {b_final:.4f}")

# 繪製結果
plt.figure(figsize=(10, 5))

# 訓練損失由 PyTorch Lightning 自動記錄
# 繪製擬合結果
X_tensor = torch.FloatTensor(X.reshape(-1, 1))
y_pred_tensor = model(X_tensor).detach().numpy()

plt.scatter(X, y, label='Data')
plt.plot(X, y_pred_tensor, 'r', label=f'y = {w_final:.2f}x + {b_final:.2f}')
plt.title('Linear Regression Fit')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()

plt.tight_layout()
plt.show()
