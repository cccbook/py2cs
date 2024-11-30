def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n - 1)

print(f'factorial(5)={factorial(5)}')
