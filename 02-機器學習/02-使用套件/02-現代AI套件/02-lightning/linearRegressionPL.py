import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# 資料
x = torch.tensor([0, 1, 2, 3, 4], dtype=torch.float32).view(-1, 1)
y = torch.tensor([1.9, 3.1, 3.9, 5.0, 6.2], dtype=torch.float32).view(-1, 1)
dataset = TensorDataset(x, y)
dataloader = DataLoader(dataset, batch_size=5, shuffle=True)

# 建立 Lightning 模型
class LinearRegressionPL(pl.LightningModule):
    def __init__(self):
        super(LinearRegressionPL, self).__init__()
        self.linear = nn.Linear(1, 1)
        self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.linear(x)
    
    def training_step(self, batch, batch_idx):
        x_batch, y_batch = batch
        y_pred = self.forward(x_batch)
        loss = self.criterion(y_pred, y_batch)
        self.log("train_loss", loss)
        return loss
    
    def configure_optimizers(self):
        return optim.SGD(self.parameters(), lr=0.01)

# 訓練模型
model = LinearRegressionPL()
trainer = pl.Trainer(max_epochs=1000, log_every_n_steps=100)
trainer.fit(model, dataloader)

# 輸出訓練後的權重與偏差
w, b = model.linear.weight.item(), model.linear.bias.item()
print(f'Trained model: y = {w:.4f}x + {b:.4f}')

# 繪製結果
x_plot = torch.linspace(0, 4, 100).view(-1, 1)
y_plot = model(x_plot).detach()
plt.scatter(x.numpy(), y.numpy(), color='red', label='Data')
plt.plot(x_plot.numpy(), y_plot.numpy(), color='blue', label='Fitted line')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()