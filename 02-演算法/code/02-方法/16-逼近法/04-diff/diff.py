from math import sin, pi

def df(f, x, h=0.00001):
    # return (f(x+h)-f(x))/h
    return (f(x+h)-f(x-h))/(2*h)

def dfn(f, x, n, h=0.00001):
    if n==0: return f(x)
    if n==1: return df(f, x, h)
    return (dfn(f, x+h, n-1, h)-dfn(f, x, n-1, h))/h

x=2
print('df(x**5, 2)=', df(lambda x:x**5, x))
for n in range(0, 10):
    print(f'dfn(x**5, 2, {n})=', dfn(lambda x:x**5, x, n))

print("====================")
print('df1=5 x**4=', 5*(x**4))
print('df2=5*4 x**3=', 5*4*(x**3))
print('df3=5*4*3 x**2=', 5*4*3*(x**2))
print('df4=5*4*3*2 x**1=', 5*4*3*2*(x**1))
print('df5=5*4*3*2*1=', 5*4*3*2*1)
