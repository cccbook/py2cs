import numpy as np
import numpy.linalg as la
import scipy.linalg as sla

a = np.random.rand(3,3)
print('a=', a)
print('det(a)=', la.det(a))

b = a.copy()
b[:, [0,1]] = b[:, [1,0]]
print('b=', b)
print('det(b)=', la.det(b)) # 兩行交換，det 變號

c = a.copy()
c[0] = c[0]*3
print('c=', c)
print('det(c)=', la.det(c)) # 一列乘以 k 倍的 det，等於 k 倍的 det 

d = a.copy()
d[1] += d[0]*3
print('d=', d)
print('det(d)=', la.det(d)) # 一列乘以 k 倍加到另一列，det 不變
