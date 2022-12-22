If  = lambda cond: lambda a: lambda b: a() if cond else b()

Fibonacci = lambda n: \
  If(n<2)(lambda:n)(lambda:Fibonacci(n-1)+Fibonacci(n-2))

print('Fibonacci(8)=', Fibonacci(8))
