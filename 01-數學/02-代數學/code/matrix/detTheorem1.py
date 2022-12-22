import numpy as np
import numpy.linalg as la
import scipy.linalg as sla

a = np.random.rand(3,3)
b = np.random.rand(3,3)
ab = np.dot(a,b)

print('det(a)=', la.det(a))
print('det(b)=', la.det(b))
print('det(a)*det(b)=', la.det(a)*la.det(b))
print('det(ab)=', la.det(ab))

