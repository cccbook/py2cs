import numpy as np
from scipy import linalg

A = np.array([[1,   -0.3], 
              [-0.1, 0.9]])
eA = linalg.eig(A)
print('eA=\n', eA)

l, X = eA
L = np.diag(l) # 把 lambda 轉成對角矩陣
print('L=\n', L)
print('X=\n', X)

XL = np.dot(X, L) # 為何用 X*L 說明在後面。 
AX = np.dot(A, X)

print('XL=\n', XL)
print('AX=\n', AX)
print('is XL==AX ?', np.allclose(XL,AX))

'''
  A1                    A1*X1 A1*X2 A1*X3
[ A2 ] [ X1 X2 X3 ] = [ A2*X1 A2*X2 A2*X3 ]
  A3                    A3*X1 A3*X2 A3*X3

A Xi = Li Xi
                             L1
A [X1 .. Xn] = [X1 ... Xn] [    L2          ]
                                  ...
'''
