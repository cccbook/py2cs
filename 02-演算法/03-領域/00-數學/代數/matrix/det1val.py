import numpy as np
a = np.ones((4,4))
for i in range(a.shape[0]): a[i,i]=3
D=np.linalg.det(a); print("D=", D)
