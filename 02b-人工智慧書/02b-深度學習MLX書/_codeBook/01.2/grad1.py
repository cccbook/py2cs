import mlx.core as mx

def f(x):
    return mx.sum(x ** 2)

x = mx.array([2.0, 3.0])
grad = mx.grad(f)(x)
print(grad)  # 輸出梯度：[4.0, 6.0]
