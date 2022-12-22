from math import sin, pi
# from dfa import df1, df2, df3, df4
from dfb import df1, df2, df3, df4

def dfn(f, x, n, h=0.00001):
    if n==0: return f(x)
    if n==1: return df1(f, x, h)
    if n==2: return df2(f, x, h)
    if n==3: return df3(f, x, h)
    if n==4: return df4(f, x, h)
    return (dfn(f, x+h, n-1, h)-dfn(f, x, n-1, h))/h

x=2
for n in range(0, 10):
    print(f'dfn(x**5, 2, {n})=', dfn(lambda x:x**5, x, n))
