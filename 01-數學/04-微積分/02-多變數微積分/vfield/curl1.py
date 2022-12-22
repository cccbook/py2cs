from vfield import *

# f = (yz, 3zx, z)
# curl(f) = (-3x, y, 2x)
def f(p):
    x,y,z = p
    return np.array([y*z, 3*z*x, z])

p = np.array([1.0, 0, 3])
print('f=(yz, 3zx, z) p=', p)
print('curl(f,p)=', curl(f,p)) # [-3,0,6]
