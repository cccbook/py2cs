import torch
import numpy as np

print("=== 基本張量創建 ===")
# 從列表創建
a = torch.tensor([1, 2, 3, 4, 5])
print("從列表創建:", a)

# 從 numpy 陣列創建
np_arr = np.array([1, 2, 3])
b = torch.from_numpy(np_arr)
print("從 numpy 創建:", b)

# 特殊矩陣
zeros = torch.zeros((2, 3))
ones = torch.ones((2, 3))
print("\n全零矩陣:\n", zeros)
print("全一矩陣:\n", ones)

# 隨機矩陣
random_matrix = torch.rand((2, 3))
print("\n隨機矩陣:\n", random_matrix)

print("\n=== 基本運算 ===")
x = torch.tensor([1, 2, 3])
y = torch.tensor([4, 5, 6])

print("加法:", x + y)
print("減法:", x - y)
print("乘法:", x * y)
print("除法:", x / y)
print("平方:", torch.square(x))
print("指數:", torch.exp(x))
print("對數:", torch.log(torch.tensor([1., 2., 3.])))  # 注意：對數需要浮點數

print("\n=== 矩陣運算 ===")
matrix1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
matrix2 = torch.tensor([[5.0, 6.0], [7.0, 8.0]])

print("矩陣乘法:\n", torch.matmul(matrix1, matrix2))
print("\n矩陣轉置:\n", torch.transpose(matrix1, 0, 1))  # 或使用 matrix1.T

print("\n=== 統計運算 ===")
data = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float)  # 使用浮點數以支持所有操作
print("平均值:", torch.mean(data))
print("總和:", torch.sum(data))
print("最大值:", torch.max(data))
print("最小值:", torch.min(data))
print("標準差:", torch.std(data))

print("\n=== 形狀操作 ===")
original = torch.tensor([[1, 2, 3], [4, 5, 6]])
print("原始形狀:", original.shape)

# 重塑
reshaped = torch.reshape(original, (3, 2))  # 或使用 original.reshape(3, 2)
print("\n重塑後 (3,2):\n", reshaped)

# 展平
flattened = torch.flatten(original)  # 或使用 original.reshape(-1)
print("\n展平後:\n", flattened)

print("\n=== 索引和切片 ===")
arr = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("原始陣列:\n", arr)
print("\n第一行:", arr[0])
print("第二列:", arr[:, 1])

print("\n=== 條件運算 ===")
condition = torch.tensor([True, False, True])
x = torch.tensor([1, 2, 3])
y = torch.tensor([4, 5, 6])
result = torch.where(condition, x, y)
print("條件選擇結果:", result)