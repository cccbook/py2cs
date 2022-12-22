import math

def diff(f, x, dx = 0.000000001) :
    dy = f(x + dx) - f(x) # dy = f(x + dx) - f(x - dx)
    return dy/dx          # return dy / (dx + dx)

print('diff(sin(x), pi/3) = ', diff(math.sin, math.pi / 3))
print('cos(pi/3) = ', math.cos(math.pi / 3))
