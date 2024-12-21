import math

def norm2(v):
    s = 0
    for x in v:
        s += x*x
    return math.sqrt(s)

def norm(v,k=2):
    s = 0
    for x in v:
        s += abs(x**k)
    return s**(1.0/k)

v = [1,1,1]
print(norm2(v))
print(norm(v,2))
print(norm(v,1))
print(norm(v,3))

# norm(v, oo) = max(|v[i]|)
#                i