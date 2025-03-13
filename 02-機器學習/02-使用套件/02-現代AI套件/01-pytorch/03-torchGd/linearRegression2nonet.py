import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

# 資料
x = torch.tensor([0, 1, 2, 3, 4], dtype=torch.float32).view(-1, 1)
y = torch.tensor([1.9, 3.1, 3.9, 5.0, 6.2], dtype=torch.float32).view(-1, 1)

# 初始化參數
w = torch.randn(1, requires_grad=True, dtype=torch.float32)
b = torch.randn(1, requires_grad=True, dtype=torch.float32)

# 優化器
optimizer = optim.SGD([w, b], lr=0.01)

# 訓練模型
num_epochs = 1000
for epoch in range(num_epochs):
    optimizer.zero_grad()
    y_pred = x * w + b  # 手動實作線性模型
    loss = F.mse_loss(y_pred, y)  # 使用 Functional API 計算 MSE Loss
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 輸出訓練後的權重與偏差
print(f'Trained model: y = {w.item():.4f}x + {b.item():.4f}')

# 繪製結果
x_plot = torch.linspace(0, 4, 100).view(-1, 1)
y_plot = x_plot * w.detach() + b.detach()
plt.scatter(x.numpy(), y.numpy(), color='red', label='Data')
plt.plot(x_plot.numpy(), y_plot.numpy(), color='blue', label='Fitted line')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
