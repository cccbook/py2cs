def df(f, x, h=0.00001):
    return (f(x+h)-f(x))/h

def dfn(f, x, n, h=0.00001):
    if n==0: return f(x)
    if n==1: return df(f, x, h)
    return (dfn(f, x+h, n-1, h)-dfn(f, x, n-1, h))/h

def factorial(n):
    r = 1
    for i in range(1,n+1):
        r = r*i
    return r

def taylor(f, x, a, k):
    r = 0
    for n in range(0, k+1):
        r += (dfn(f, a, n)/factorial(n))*(x-a)**n
    return r

for n in range(0,10):
    print(f'factorial({n})=', factorial(n))

x = 2.2; a = 2.0
print('x=', 2.2, 'a=', a)
print('x^2=', x**2)
print('taylor(x*x, x=2.2, a=2, k=4)=', taylor(lambda x:x*x, x, a, 4))
print('2.2^3=', 2.2**3)
print('taylor(x**3, x=2.2, a=2, k=3)=', taylor(lambda x:x**3, x, a, 3))
print('taylor(x**3, x=2.2, a=2, k=5)=', taylor(lambda x:x**3, x, a, 5)) # 多取幾階反而爛掉了！

print('2.2^10=', 2.2**10)
print('taylor(x**10, x=2.2, a=2, k=12)=', taylor(lambda x:x**10, x, a, 12))
