If = lambda cond, a, b: a if cond else b

def Fibonacci(n):
    If(n<2, n, Fibonacci(n-1)+Fibonacci(n-2))

print('Fibonacci(8)=', Fibonacci(8))
