def abs(x):
    return x if x >= 0 else -x

def square(x):
    return x*x

def sqrt_iter(guess, x):
    return guess if is_good_enough(guess, x) else sqrt_iter(improve(guess, x), x)

def improve(guess, x):
    return average(guess, x / guess)

def average(x, y):
    return (x + y) / 2

def is_good_enough(guess, x):
    return abs(square(guess) - x) < 0.001

def sqrt(x):
    return sqrt_iter(1, x)

print('sqrt(9)=', sqrt(9))
print('sqrt(100 + 37)=', sqrt(100 + 37))
print('sqrt(sqrt(2) + sqrt(3))=', sqrt(sqrt(2) + sqrt(3)))
print('square(sqrt(1000))=', square(sqrt(1000)))