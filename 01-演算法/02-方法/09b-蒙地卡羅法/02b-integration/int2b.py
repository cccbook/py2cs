import numpy as np
import random

def f(x,y):
    return x**2+2*y**2

step = 0.001

def integrate(f, rx, ry):
    area = 0.0
    for x in np.arange(rx[0], rx[1], step):
        for y in np.arange(ry[0], ry[1], step):
            area += f(x,y)*step*step
    return area

print(integrate(f, [0,1], [0,1]))

upper = 10
lower = 0

def mcInt(f, rfrom, rto, n=100000):
    hits = 0
    for i in range(0, n):
        x = random.uniform(rfrom[0], rto[0])
        y = random.uniform(rfrom[1], rto[1])
        fz = random.uniform(0, upper)
        if f(x,y)>fz:
            hits += 1
    return (rto[0]-rfrom[0])*(rto[1]-rfrom[1])*(upper-lower)*hits/n

print(mcInt(f, [0,0], [1,1]))
