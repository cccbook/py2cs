import numpy as np
from scipy import linalg

m,n=500,50

A = np.random.rand(m, m)
B = np.random.rand(m, n)
print('A=', A)
print('B=', B)
X1 = linalg.solve(A,B)
X2 = np.dot(linalg.inv(A), B)
print('X1=solve(A,B)=', X1)
print('X2=inv(A)*B', X2)
print('is X1==X2 ?', np.allclose(X1,X2))

# %timeit linalg.solve(A,B)
# %timeit np.dot(linalg.inv(A), B)
