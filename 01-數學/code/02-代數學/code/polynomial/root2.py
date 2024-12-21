import math

def root2(a,b,c):
    t = b*b - 4*a*c
    if (t < 0):
        raise Exception('沒有實根')
    t2 = math.sqrt(t)
    return [(-b+t2)/(2*a), (-b-t2)/(2*a)]


print("root of 1x^2+4x+0=", root2(1,4,0))
