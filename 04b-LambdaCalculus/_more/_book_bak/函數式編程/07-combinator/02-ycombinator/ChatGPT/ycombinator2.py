Y = lambda g: g(lambda: Y(g))

factorial = Y(lambda f: lambda n: 1 if n == 0 else n * f()(n - 1))

print(factorial(5))  # 輸出 120
