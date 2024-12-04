If = lambda cond, a, b: a if cond else b

def Fibonacci(n):
    r = Fibonacci(n-1)+Fibonacci(n-2)
    return If(n<2, n, r)

print('Fibonacci(8)=', Fibonacci(8))
