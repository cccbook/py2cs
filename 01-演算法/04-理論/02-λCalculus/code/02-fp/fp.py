def RANGE(m,n):
    r = []
    for i in range(m,n+1):
        r.append(i)
    return r

def EACH(a, f):
    for x in a:
        f(x)

def MAP(a, f):
    r = []
    for x in a:
        r.append(f(x))
    return r

def FILTER(a, f):
    r = []
    for x in a:
        if f(x): r.append(x)
    return r

def REDUCE(a, f, init):
    r = init
    for x in a:
        r = f(r, x)
    return r

if __name__=="__main__":
    a = RANGE(1,5)
    EACH(a, lambda x:print(x))
    print(MAP(a, lambda x:x*x))
    print(FILTER(a, lambda x:x%2==1))
    print(REDUCE(a, lambda x,y:x+y, 0))
