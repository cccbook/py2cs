# https://chatgpt.com/c/67593caf-f8bc-8012-8c20-c01eb1da1c79
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 設定隨機種子，保證結果可重現
np.random.seed(42)
torch.manual_seed(42)

# 生成訓練數據
def generate_data(num_samples=10000):
    # 生成隨機的 32 位元整數對 (0-2^32-1)
    x1 = np.random.randint(0, 2**32, size=num_samples)
    x2 = np.random.randint(0, 2**32, size=num_samples)
    # 計算加法結果
    y = x1 + x2
    # 將數據轉換為適當的形狀
    return np.column_stack((x1, x2)), y

# 生成訓練數據
X_train, y_train = generate_data(10000)
X_test, y_test = generate_data(1000)

# 將數據轉換為 PyTorch 張量
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)  # 轉換為列向量
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# 定義模型
class AdderModel(nn.Module):
    def __init__(self):
        super(AdderModel, self).__init__()
        self.fc1 = nn.Linear(2, 64)  # 2 個輸入（32 位元的兩個數字）
        self.fc2 = nn.Linear(64, 64) # 隱藏層：64 個神經元
        self.fc3 = nn.Linear(64, 1)  # 輸出層：1 個神經元，對應加法結果

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# 初始化模型
model = AdderModel()

# 定義損失函數和優化器
criterion = nn.MSELoss()  # 均方誤差損失函數
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 訓練模型
epochs = 100000
for epoch in range(epochs):
    model.train()
    
    # 前向傳播
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    
    # 反向傳播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 2 == 0:  # 每 2 次訓練輸出一次
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# 在測試數據上評估模型
model.eval()  # 切換到評估模式
with torch.no_grad():
    test_outputs = model(X_test)
    test_loss = criterion(test_outputs, y_test)
    print(f"Test Loss: {test_loss.item():.4f}")
