import optimize as opt
import random

def improve(a):
    i = random.randrange(0, len(a)-1)
    if (a[i] > a[i+1]):
        a[i], a[i+1] = a[i+1], a[i]
        return a

def test_improving():
    x = [3,5,1,2,4]
    print('init:x=', x)
    x = opt.improving(x, improve)
    print('best:x=', x)
    assert x == [1,2,3,4,5]
