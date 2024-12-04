If = lambda cond, a, b: a if cond else b

def out(x):
    print(x)

Fibonacci = lambda n: \
  If(n<2, out(n), Fibonacci(n-1)+Fibonacci(n-2))

print('Fibonacci(8)=', Fibonacci(8))