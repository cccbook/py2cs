import mlx.core as mx

# 創建 2x2 張量，並啟用自動微分
# x = mx.array([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
x = mx.array([[1.0, 2.0], [3.0, 4.0]]

# 定義一個矩陣運算：y = x^2 + 2x + 1
y = x**2 + 2*x + 1

# 前向傳播：計算 y
print("y:\n", y)

# 反向傳播：計算梯度
y.backward()

# 查看 x 的梯度
print("Gradient of x:\n", x.grad)
