def each(a, f):
    for x in a:
        f(x)

def map(a, f):
    r = []
    for x in a:
        r.append(f(x))
    return r

def filter(a, f):
    r = []
    for x in a:
        if f(x): r.append(x)
    return r

def reduce(a, f, init):
    r = init
    for x in a:
        r = f(r, x)
    return r

if __name__=="__main__":
    a = range(1,5)
    each(a, lambda x:print(x))
    each(a, lambda x:print(x) if x%2==0 else None)
    print(map(a, lambda x:x*x))
    print(filter(a, lambda x:x%2==1))
    print(reduce(a, lambda x,y:x+y, 0))
