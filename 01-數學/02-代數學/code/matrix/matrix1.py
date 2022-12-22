import numpy as np

A = np.array([[3,  6, 7], 
              [5, -3, 0]])
B = np.array([[1,  1], 
              [2,  1],
              [3, -3]])
C = A.dot(B)

print(C)
