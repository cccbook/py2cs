def pmul(a,b):
    c = [0] * (len(a)+len(b)-1)
    for i in range(len(a)):
        for j in range(len(b)):
            c[i+j] += a[i]*b[j]
    return c

def pdiv(a,b):
    # print('a=', a, 'b=', b)
    la, lb = len(a), len(b)
    r = a.copy()
    q = [0] * (la-lb+1)
    for i in range(la-lb+1):
        scale = r[i]/b[0]
        q[i] = scale
        for j in range(lb):
            r[i+j] -= b[j]*scale
        # print('r=', r)
    return :'r': r[-lb+1:], 'q': q

def pzero(a, epsilon=0.00001):
    for x in a:
        if abs(x) > epsilon: return False
    return True

def pgcd(a,b):
    while not pzero(b):
        d = pdiv(a,b)
        print('d=', d)
        a = b
        b = d['r']
    return a

print('pmul([1,2],[1,1])=', pmul([1,2],[1,1]))
print('pdiv([3,3,2],[1,1])=', pdiv([3,3,2],[1,1]))
print('pdiv([1,3,2],[1,1,1])=', pdiv([1,3,2],[1,1,1]))
print('pdiv([1,4,3],[1,3,2])=', pdiv([1.0,4.0,3.0],[1.0,3.0,2.0]))


x = [1,1]
y = [1,2]
z = [1,3]
xy = pmul(x,y)
xz = pmul(x,z)
print('xy = ', xy)
print('xz = ', xz)
print('gcd(xz, xy) = ', pgcd(xz, xy))

