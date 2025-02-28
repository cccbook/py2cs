import torch
import torch.nn as nn
import torch.optim as optim

# 七段顯示器的真值表（輸入）
seven_segment_inputs = torch.tensor([
    [1, 1, 1, 1, 1, 1, 0],  # 0
    [0, 1, 1, 0, 0, 0, 0],  # 1
    [1, 1, 0, 1, 1, 0, 1],  # 2
    [1, 1, 1, 1, 0, 0, 1],  # 3
    [0, 1, 1, 0, 0, 1, 1],  # 4
    [1, 0, 1, 1, 0, 1, 1],  # 5
    [1, 0, 1, 1, 1, 1, 1],  # 6
    [1, 1, 1, 0, 0, 0, 0],  # 7
    [1, 1, 1, 1, 1, 1, 1],  # 8
    [1, 1, 1, 1, 0, 1, 1],  # 9
], dtype=torch.float32)

# 目標輸出（數字的 4 位元二進位）
binary_outputs = torch.tensor([
    [0, 0, 0, 0],  # 0
    [0, 0, 0, 1],  # 1
    [0, 0, 1, 0],  # 2
    [0, 0, 1, 1],  # 3
    [0, 1, 0, 0],  # 4
    [0, 1, 0, 1],  # 5
    [0, 1, 1, 0],  # 6
    [0, 1, 1, 1],  # 7
    [1, 0, 0, 0],  # 8
    [1, 0, 0, 1],  # 9
], dtype=torch.float32)

# 定義 MLP 模型
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(7, 16)   # 7 -> 16 隱藏層
        self.fc2 = nn.Linear(16, 8)   # 16 -> 8 隱藏層
        self.fc3 = nn.Linear(8, 4)    # 8 -> 4 輸出層
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))  # 用 sigmoid 限制輸出在 0~1 之間
        return x

# 初始化模型
model = MLP()

# 定義損失函數和優化器
criterion = nn.MSELoss()  # 均方誤差
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 訓練模型
epochs = 5000
for epoch in range(epochs):
    optimizer.zero_grad()  # 清空梯度
    outputs = model(seven_segment_inputs)  # 前向傳播
    loss = criterion(outputs, binary_outputs)  # 計算損失
    loss.backward()  # 反向傳播
    optimizer.step()  # 更新權重

    # 每 500 次顯示損失
    if (epoch + 1) % 500 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}')

# 測試預測
def predict(segment_input):
    with torch.no_grad():
        output = model(torch.tensor(segment_input, dtype=torch.float32))
        binary_result = (output > 0.5).int()  # 轉換為 0 或 1
        return "".join(map(str, binary_result.numpy()))

print("\n=== 測試結果 ===")
for i, segment in enumerate(seven_segment_inputs):
    print(f"Input: {segment.numpy()} -> Predicted: {predict(segment)}")
