import mlx.core as mx
import numpy as np

print("=== 基本張量創建 ===")
# 從列表創建
a = mx.array([1, 2, 3, 4, 5])
print("從列表創建:", a)

# 從 numpy 陣列創建
np_arr = np.array([1, 2, 3])
b = mx.array(np_arr)
print("從 numpy 創建:", b)

# 特殊矩陣
zeros = mx.zeros((2, 3))
ones = mx.ones((2, 3))
print("\n全零矩陣:\n", zeros)
print("全一矩陣:\n", ones)

# 隨機矩陣
random_matrix = mx.random.uniform(shape=(2, 3))
print("\n隨機矩陣:\n", random_matrix)

print("\n=== 基本運算 ===")
x = mx.array([1, 2, 3])
y = mx.array([4, 5, 6])

print("加法:", x + y)
print("減法:", x - y)
print("乘法:", x * y)
print("除法:", x / y)
print("平方:", mx.square(x))
print("指數:", mx.exp(x))
print("對數:", mx.log(mx.array([1, 2, 3])))

print("\n=== 矩陣運算 ===")
matrix1 = mx.array([[1.0, 2.0], [3.0, 4.0]])
matrix2 = mx.array([[5.0, 6.0], [7.0, 8.0]])

print("矩陣乘法:\n", mx.matmul(matrix1, matrix2))
print("\n矩陣轉置:\n", mx.transpose(matrix1))

print("\n=== 統計運算 ===")
data = mx.array([1, 2, 3, 4, 5])
print("平均值:", mx.mean(data))
print("總和:", mx.sum(data))
print("最大值:", mx.max(data))
print("最小值:", mx.min(data))
print("標準差:", mx.std(data))

print("\n=== 形狀操作 ===")
original = mx.array([[1, 2, 3], [4, 5, 6]])
print("原始形狀:", original.shape)

# 重塑
reshaped = mx.reshape(original, (3, 2))
print("\n重塑後 (3,2):\n", reshaped)

# 展平
flattened = mx.reshape(original, (-1,))
print("\n展平後:\n", flattened)

print("\n=== 索引和切片 ===")
arr = mx.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("原始陣列:\n", arr)
print("\n第一行:", arr[0])
print("第二列:", arr[:, 1])

print("\n=== 條件運算 ===")
condition = mx.array([True, False, True])
x = mx.array([1, 2, 3])
y = mx.array([4, 5, 6])
result = mx.where(condition, x, y)
print("條件選擇結果:", result)