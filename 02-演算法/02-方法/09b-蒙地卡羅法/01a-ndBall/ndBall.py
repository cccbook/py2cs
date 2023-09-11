import numpy as np
from math import pi, gamma

def mcBallVolume(n, r, tries=100000):
    count = 0
    for i in range(tries):
        vector = 2*r*np.random.random_sample(n)-r
        if np.linalg.norm(vector, 2) <= r:
            count += 1
    return (count/tries)*((2*r)**n)

def ballVolume(n, r):
    return pi**(n/2)/gamma(n/2+1) * (r**n)


r = 1
print('r=', r)

print(f'ballVolume(2d, r)=', ballVolume(2, r))
print('pi r^2=', pi*(r**2))
print(f'mcBallVolume(2d, r)=', mcBallVolume(2, r))

print(f'ballVolume(3d, r)=', ballVolume(3, r))
print('4/3 pi r^3=', 4/3*pi*(r**3))
print(f'mcBallVolume(3d, r)=', mcBallVolume(3, r))

print(f'ballVolume(4d, r)=', ballVolume(4, r))
print(f'mcBallVolume(4d, r)=', mcBallVolume(4, r))
