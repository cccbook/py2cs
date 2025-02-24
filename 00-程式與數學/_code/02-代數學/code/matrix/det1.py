import numpy as np
 
mat = [[1, 0, 2, -1],
       [3, 0, 0, 5],
       [2, 1, 4, -3],
       [1, 0, 5, 0]]
print('mat=', mat)
print('det=', np.linalg.det(mat))
