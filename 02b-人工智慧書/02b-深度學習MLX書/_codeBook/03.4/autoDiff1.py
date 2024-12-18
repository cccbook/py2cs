import mlx.core as mx

# 創建張量並啟用自動微分
# x = mx.array([2.0, 3.0], requires_grad=True)
x = mx.array([2.0, 3.0])
y = x**2 + 3*x + 5  # 函數：y = x^2 + 3x + 5

# 前向傳播：計算 y
print("y:", y)

# 反向傳播：計算梯度
y.backward()

# 查看 x 的梯度
print("Gradient of x:", x.grad)
