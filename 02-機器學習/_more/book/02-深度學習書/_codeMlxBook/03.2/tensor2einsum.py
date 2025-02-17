import mlx.core as mx

# 從列表創建張量
tensor = mx.array([[1, 2, 3], [4, 5, 6]])
print("Tensor:\n", tensor)

# 從 NumPy 數組創建張量
import numpy as np
np_array = np.array([[1, 2], [3, 4]])
mlx_tensor = mx.array(np_array)
print("MLX Tensor:\n", mlx_tensor)

# 指定數據類型
#float_tensor = mx.array([1, 2, 3], dtype="float32")
#print("Float Tensor:", float_tensor)

a = mx.array([[1, 2], [3, 4]])
b = mx.array([[5, 6], [7, 8]])

# 加法
add_result = a + b
print("Add:\n", add_result)

# 乘法（逐元素）
mul_result = a * b
print("Multiply:\n", mul_result)

# 指數運算
exp_result = mx.exp(a)
print("Exponential:\n", exp_result)

# 矩陣乘法
a = mx.array([[1.0, 2], [3, 4]])
b = mx.array([[5, 6], [7, 8]])

matmul_result = mx.matmul(a, b)
print("Matrix Multiplication:\n", matmul_result)

# 轉置
transposed = mx.transpose(a)
print("Transposed Matrix:\n", transposed)

a = mx.array([[1.0, 2], [3, 4]])
b = mx.array([[5.0, 6], [7, 8]])

# 使用 mlx.einsum 實現矩陣乘法
result = mx.einsum("ij,jk->ik", a, b)
print("Matrix Multiplication with einsum:\n", result)

x = mx.array([1.0, 2, 3])
y = mx.array([4.0, 5, 6])

# 內積（向量點積）
dot_product = mx.einsum("i,i->", x, y)
print("Dot Product:", dot_product)

a = mx.array([[1, 2, 3], [4, 5, 6]])

# 使用 mlx.einsum 實現轉置
transposed = mx.einsum("ij->ji", a)
print("Transposed Tensor with einsum:\n", transposed)

A = mx.random.uniform(shape=(3, 2, 4))
B = mx.random.uniform(shape=(3, 4, 5))

# 批次矩陣乘法
batch_result = mx.einsum("bij,bjk->bik", A, B)
print("Batch Matrix Multiplication Result Shape:", batch_result.shape)

a = mx.array([[1, 2, 3], [4, 5, 6]])
b = mx.array([1, 2, 3])  # 廣播成與 a 相同的形狀

# 廣播相加
broadcasted_result = a + b
print("Broadcasted Addition:\n", broadcasted_result)

tensor = mx.array([[1, 2, 3], [4, 5, 6]])

# 索引單個元素
print("Element at (0, 1):", tensor[0, 1])

# 切片操作
print("First Row:", tensor[0])
print("First Column:", tensor[:, 0])

