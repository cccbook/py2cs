import torch

# 定義變數，並啟用梯度計算
x = torch.tensor(0.0, requires_grad=True)
y = torch.tensor(0.0, requires_grad=True)
z = torch.tensor(0.0, requires_grad=True)

# 設定優化器
optimizer = torch.optim.SGD([x, y, z], lr=0.1)  # 使用隨機梯度下降法，學習率 0.1

# 進行梯度下降
for _ in range(100):  # 迭代 100 次
    optimizer.zero_grad()  # 清除上一次的梯度
    f = x**2 + y**2 + z**2 - 2*x - 4*y - 6*z + 8  # 目標函數
    f.backward()  # 計算梯度
    optimizer.step()  # 更新變數

# 輸出最小值點
print(f"Minimum at x = {x.item()}, y = {y.item()}, z = {z.item()}")
