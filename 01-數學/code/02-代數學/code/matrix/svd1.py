import numpy as np

a = [[1,2,3], [4,5,6]]

u,s,vh = np.linalg.svd(a, full_matrices=True)

print('u=', u, '\ns=', s, '\nvh=', vh)

ds = np.diag(s)
print('ds=', ds)
ds1 = np.append(ds, [[0],[0]], 1)
print('ds1=', ds1)
us = np.dot(u, ds1)
usvh = np.dot(us, vh)
print('usvh=', usvh)

print('is usvh==a ?', np.allclose(usvh,a))
