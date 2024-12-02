from vfield import *
from math import *

def F(p):
    x,y=p
    return np.array([-y, -x*y])

def r(t):
    return np.array([cos(t), sin(t)])

print("lintegrate(0,pi/2)=", lintegrate(F, r, 0., pi/2))
