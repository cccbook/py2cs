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

def mcInt(f, rx, ry, n=100000):
    hits = 0
    for i in range(0, n):
        x = random.uniform(rx[0], rx[1])
        y = random.uniform(ry[0], ry[1])
        fz = random.uniform(0, upper)
        if f(x,y)>fz:
            hits += 1
    return (rx[1]-rx[0])*(ry[1]-ry[0])*(upper-lower)*hits/n

print(mcInt(f, [0,1], [0,1]))
