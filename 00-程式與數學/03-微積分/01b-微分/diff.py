import math

def diff(f, x, dx = 0.0001) :
    print('f(x)=', f(x))
    print('f(x+dx)=', f(x+dx))
    dy = f(x + dx) - f(x) # dy = f(x + dx) - f(x - dx)
    return dy/dx          # return dy / (dx + dx)

if __name__=="__main__":
    print('diff(sin(x), pi/3) = ', diff(math.sin, math.pi / 3))
    print('cos(pi/3) = ', math.cos(math.pi / 3))
