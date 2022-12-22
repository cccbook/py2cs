def f(x) :
    return x**2-4*x+1

def bsolve(f,a,b):
    c = (a+b)/2 
    if abs(a-b) < 0.00001:
        return c 
    if f(c)*f(a) >= 0:
        return bsolve(f, c, b) 
    else:
        return bsolve(f, a, c) 


x=bsolve(f, 0, 1) 
print("x=", x, " f(x)=", f(x)) 
