import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# 資料
x = torch.tensor([0, 1, 2, 3, 4], dtype=torch.float32).view(-1, 1)
y = torch.tensor([1.9, 3.1, 3.9, 5.0, 6.2], dtype=torch.float32).view(-1, 1)

# 初始化參數
w = torch.randn(1, dtype=torch.float32, requires_grad=True)
b = torch.randn(1, dtype=torch.float32, requires_grad=True)

# 訓練模型
num_epochs = 1000
learning_rate = 0.01
for epoch in range(num_epochs):
    y_pred = x * w + b  # 手動實作線性模型
    loss = F.mse_loss(y_pred, y)  # 使用 Functional API 計算 MSE Loss
    
    loss.backward()
    
    # 手動更新參數
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad
        
        # 清空梯度
        w.grad.zero_()
        b.grad.zero_()
    
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