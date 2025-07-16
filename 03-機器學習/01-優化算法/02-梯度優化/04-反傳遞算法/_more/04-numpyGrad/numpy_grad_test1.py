from numpy_grad_deepseek import *

# 創建一個簡單的模型
model = Linear(2, 3)
x = Tensor([[1, 2], [3, 4]], requires_grad=True)

# 前向傳播
y = model(x)
z = y.relu()

# 反向傳播
z.backward()

# 輸出結果
print("Input:", x.data)
print("Output:", z.data)
print("Gradient of x:", x.grad)
print("Gradient of weights:", model.weights.grad)
print("Gradient of bias:", model.bias.grad)