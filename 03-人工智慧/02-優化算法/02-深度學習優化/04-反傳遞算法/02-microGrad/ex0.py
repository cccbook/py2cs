from micrograd.engine import Value

x = Value(1.0)
y = Value(2.0)
z = Value(3.0)
q = x*y
f = q+z
# print(f'{f.data:.4f}')
f.backward()
print(f'gx={x.grad:.4f}')
print(f'gy={y.grad:.4f}')
print(f'gz={z.grad:.4f}')