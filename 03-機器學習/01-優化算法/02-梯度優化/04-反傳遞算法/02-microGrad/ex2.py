from micrograd.engine import Value

x = Value(0.0)
y = Value(0.0)
z = Value(0.0)
loss = x**2+y**2+z**2-2*x-4*y-6*z+8
loss.backward()
print(f'gx={x.grad:.4f}')
print(f'gy={y.grad:.4f}')
print(f'gz={z.grad:.4f}')