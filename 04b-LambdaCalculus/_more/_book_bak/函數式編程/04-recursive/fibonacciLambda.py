fibonacci = lambda n: \
    0 if n == 0 else  \
    1 if n == 1 else  \
    fibonacci(n-1) + fibonacci(n-2)

print('fibonacci(8)=', fibonacci(8))
