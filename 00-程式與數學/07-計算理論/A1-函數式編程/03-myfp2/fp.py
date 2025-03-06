def _each(a, f, i):
    if i < len(a):
        f(a[i])
        _each(a, f, i+1)

def each(a, f):
    _each(a, f, 0)

def _map(a, f, i, r):
    if i < len(a):
        r.append(f(a[i]))
        _map(a, f, i+1, r)

def map(a, f):
    r = []
    _map(a, f, 0, r)
    return r

def _filter(a, f, i, r):
    if i < len(a):
        if f(a[i]): r.append(a[i])
        _filter(a, f, i+1, r)

def filter(a, f):
    r = []
    _filter(a, f, 0, r)
    return r

def _reduce(a, f, i, r):
    if i < len(a):
        r = f(r, a[i])
        return _reduce(a, f, i+1, r)
    else:
        return r

def reduce(a, f, init):
    r = init
    return _reduce(a, f, 0, r)

if __name__=="__main__":
    a = range(1,5)
    each(a, lambda x:print(x))
    each(a, lambda x:print(x) if x%2==0 else None)
    print(map(a, lambda x:x*x))
    print(filter(a, lambda x:x%2==1))
    print(reduce(a, lambda x,y:x+y, 0))
