import math

def sqrt(x):
    if x < 0:
        return 1j*math.sqrt(-x)
    else:
        return math.sqrt(x)

def root2(a,b,c):
    t = b*b - 4*a*c
    t2 = sqrt(t)
    return [(-b+t2)/(2*a), (-b-t2)/(2*a)]


r1, r2 = root2(2,1,5)
print("root of 2x^2+1x+5: r1=", r1, "r2=", r2)
print("2*r1**2+r1+5=", 2*r1**2+r1+5)
print("2*r2**2+r2+5=", 2*r2**2+r2+5)
