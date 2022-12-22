def isClose(op, a, b):
    return isinstance(op(a,b), type(a))

def isAssoc(op, a, b, c, eq):
    return eq(op(op(a,b), c), op(a, op(b,c)))

def isId(op, e, a, eq):
    return eq(op(a,e), a)

def isInv(op, e, a, b, eq):
    return eq(op(a,b), e)

def isGroup(op, gen, inv, e, eq, times=1000):
    for _ in range(times):
        isFail = False
        a,b,c=gen(),gen(),gen()
        if not isClose(op, a, b):
            print('not isClose')
            isFail = True
        if not isAssoc(op, a, b, c, eq):
            print('not isAssoc')
            isFail = True
        if not isId(op, e, a, eq):
            print('not isId')
            isFail = True
        if not isInv(op, e, a, inv(a), eq):
            print('not isInverse')
            isFail = True
        if isFail:
            print('a=', a, 'b=', b, 'c=', c)
            return

import random

def iadd(a,b):
    return a+b

def iinv(a):
    return -a

def igen():
    return random.randint(-1000000, 1000000)

def ieq(a, b):
    return a==b

def fadd(a,b):
    return a+b

def finv(a):
    return -a

def fgen():
    return random.uniform(-1000000.0, 1000000.0)

def feq(a,b):
    return abs(a-b) < 0.0001

def fsadd(a,b):
    return lambda x:a(x)+b(x)

def fsinv(a):
    return lambda x:-a(x)

#def fsgen():
#    return random.uniform(-1000000.0, 1000000.0) 

def fseq(a,b):
    x = random.uniform(-1000000.0, 1000000.0)
    return a(x)==b(x)

isGroup(iadd, igen, iinv, 0, ieq)
isGroup(fadd, fgen, finv, 0.0, feq)
