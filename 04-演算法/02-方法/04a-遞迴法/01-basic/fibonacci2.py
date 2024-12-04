from datetime import datetime

def fibonacci (n):
    if n < 0: raise
    if n == 0: return 0
    if n == 1: return 1
    print(f'calculate f({n})')
    f1 = fibonacci(n - 1)
    print('f1=', f1)
    f2 = fibonacci(n - 2)
    print('f2=', f2)
    return f1+f2


n = 4
startTime = datetime.now()
print(f'fibonacci({n})={fibonacci(n)}')
endTime = datetime.now()
seconds = endTime - startTime
print(f'time:{seconds}')
