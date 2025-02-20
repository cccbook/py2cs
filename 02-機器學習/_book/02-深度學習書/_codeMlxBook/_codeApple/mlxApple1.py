import mlx.core as mx
a = mx.array([1, 2, 3, 4])
print('a=', a)
print('a.shape=', a.shape)
print('a.dtype=', a.dtype)

b = mx.array([1.0, 2.0, 3.0, 4.0])
print('b=', a)
print('b.shape=', a.shape)
print('b.dtype=', a.dtype)

c = a + b    # c not yet evaluated
mx.eval(c)   # evaluates c
print('c=', c, 'after eval()')

c = a + b
print(c)     # 在 print(c) 時會 evaluates c

c = a + b
import numpy as np
print('np.array(c)=', np.array(c))   # 在 np.array(c) 時會 evaluates c

x = mx.array(0.0)

print('sin(x)=', mx.sin(x))

print('grad(sin)(x)=', mx.grad(mx.sin)(x))

print('grad(grad(sin))(x)=', mx.grad(mx.grad(mx.sin))(x))
