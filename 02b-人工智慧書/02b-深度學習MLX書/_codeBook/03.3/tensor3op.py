import mlx.core as mx

# 創建兩個矩陣
A = mx.array([[1.0, 2, 3], [4, 5, 6]])
B = mx.array([[7.0, 8, 9], [10, 11, 12]])

# 矩陣加法
C = A + B
print("Matrix Addition:\n", C)

# 逐元素相乘
D = A * B
print("Element-wise Multiplication:\n", D)

# 矩陣乘法
#E = mx.matmul(A, mx.transpose(B))
#print("Matrix Multiplication:\n", E)

# 矩陣轉置
B_T = mx.transpose(B)
print("Transposed Matrix B:\n", B_T)

# 綜合運算：A * (B 的轉置) + C
F = mx.matmul(A, B_T) # + C
print("Combined Operation Result:\n", F)

# 廣播加法
A = mx.array([[1, 2, 3], [4, 5, 6]])
b = mx.array([10, 20, 30])  # 與 A 的每一行進行加法

broadcast_add = A + b
print("Broadcast Addition:\n", broadcast_add)

