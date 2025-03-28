import mlx.core as mx

# 創建一個一維張量（向量）
tensor_1d = mx.array([1, 2, 3])
print(tensor_1d)

# 創建一個二維張量（矩陣）
tensor_2d = mx.array([[1, 2, 3], [4, 5, 6]])
print(tensor_2d)

# 全零張量
zeros = mx.zeros((2, 3))
print("Zeros Tensor:\n", zeros)

# 全一張量
ones = mx.ones((2, 3))
print("Ones Tensor:\n", ones)

# 隨機張量
random_tensor = mx.random.uniform(shape=(2, 3))
print("Random Tensor:\n", random_tensor)

tensor = mx.array([[1, 2, 3], [4, 5, 6]])

# 張量的形狀（Shape）
print("Shape:", tensor.shape)  # (2, 3)

# 張量的維度數（Rank）
print("Rank:", len(tensor.shape))  # 2

# 張量的數據類型（Data Type）
print("Data Type:", tensor.dtype)  # mlx.core.int32（默認）

a = mx.array([[1, 2], [3, 4]])
b = mx.array([[5, 6], [7, 8]])

# 加法
add_result = a + b
print("Add:\n", add_result)

# 減法
sub_result = a - b
print("Subtract:\n", sub_result)

# 乘法（逐元素）
mul_result = a * b
print("Multiply:\n", mul_result)

# 除法（逐元素）
div_result = a / b
print("Divide:\n", div_result)

tensor = mx.array([[1, 2, 3], [4, 5, 6]])

# 重塑張量
reshaped_tensor = mx.reshape(tensor, (3, 2))
print("Reshaped Tensor:\n", reshaped_tensor)

# 轉置張量
transposed_tensor = mx.transpose(tensor)
print("Transposed Tensor:\n", transposed_tensor)


tensor = mx.array([[1, 2, 3], [4, 5, 6]])

# 總和
sum_tensor = mx.sum(tensor)
print("Sum:", sum_tensor)

# 沿維度 0 求和（列求和）
sum_dim0 = mx.sum(tensor, axis=0)
print("Sum along axis 0:\n", sum_dim0)

# 平均值
mean_tensor = mx.mean(tensor)
print("Mean:", mean_tensor)

# L2 範數 -- fail
# l2_norm = mx.norm(tensor)
# print("L2 Norm:", l2_norm)

# 指數與對數操作
exp_tensor = mx.exp(tensor)
print("Exponential:\n", exp_tensor)

log_tensor = mx.log(tensor)
print("Logarithm:\n", log_tensor)

import numpy as np

# Numpy 轉 MLX 張量
np_array = np.array([[1, 2], [3, 4]])
mlx_tensor = mx.array(np_array)
print("MLX Tensor:\n", mlx_tensor)

# MLX 張量轉 Numpy -- fail
# converted_np_array = mlx_tensor.numpy()
# print("Numpy Array:\n", converted_np_array)

a = mx.random.uniform(shape=(1000, 1000))
b = mx.random.uniform(shape=(1000, 1000))

result = a + b
print("Computation Completed!")

