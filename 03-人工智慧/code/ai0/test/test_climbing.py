import random
import numpy as np
import optimize as opt
from math0 import near

def height(x):
    return -1*(x*x)

def neighbor(x, dx=0.01):
    step = random.choice([-dx, dx])
    x += step
    return x

def test_climbing():
    x = 0.0
    print('init:x=', x)
    x = opt.climbing(x, neighbor, height)
    print('best:x=', x)
    assert near(x, 0.0)