Fibonacci = lambda n: \
  n if n<2 else Fibonacci(n-1)+Fibonacci(n-2)

print('Fibonacci(8)=', Fibonacci(8))

