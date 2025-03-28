import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 資料
x = torch.tensor([0, 1, 2, 3, 4], dtype=torch.float32).view(-1, 1)
y = torch.tensor([1.9, 3.1, 3.9, 5.0, 6.2], dtype=torch.float32).view(-1, 1)
# 上述程式的 view(-1,1) 請參考 
# 1. https://chatgpt.com/share/67d25213-102c-8012-8569-ed13ec8f3c99
# 2. https://pytorch.org/docs/stable/generated/torch.Tensor.view.html

# 建立簡單的線性回歸模型
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)  # 輸入維度 1，輸出維度 1
        # nn.Linear 裡面有 bias 參數，所以 weight 維度為 1
        # 參考 -- https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
    
    def forward(self, x):
        return self.linear(x)

# 初始化模型、損失函數與優化器
model = LinearRegression()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 訓練模型
num_epochs = 1000
for epoch in range(num_epochs):
    optimizer.zero_grad()
    y_pred = model(x)
    loss = criterion(y_pred, y)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

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
