import mlx.core as mx

x = mx.array([[1, 2], [3, 4]])
y = mx.array([[5, 6], [7, 8]])
z = mx.dot(x, y)  # 矩陣乘法
print(z)
