import numpy as np
import numpy.linalg as la
import scipy.linalg as sla

a = np.random.rand(3,3)
p,l,u = sla.lu(a)
print('a=', np.dot(p, np.dot(l,u)))
print('p*l*u=', np.dot(p, np.dot(l,u)))
print('u =', u)
print('det(a)=', la.det(a))
print('det(u)=', la.det(u))
print('det(l)=', la.det(l))
print('det(p)=', la.det(p))

