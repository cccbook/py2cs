from vfield import *

# f = (3xz, 2xy, -yzz)
# div(f) = (3z+2x-2yz)
def f(p):
    x,y,z = p
    return np.array([3*x*z, 2*x*y, -y*z*z])

p = np.array([1.0, 0, 3])
print('div(f,p)=', div(f,p)) # 3z+2x+0 = 3*3+2*1+0 = 11
