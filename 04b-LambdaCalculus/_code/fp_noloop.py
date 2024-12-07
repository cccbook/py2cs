def _each(a, f, i):
    if i==len(a):
        return
    else:
        f(a[i])
        _each(a, f, i+1)

def EACH(a, f):
    _each(a,f,0)

def _map(a, f, i, r):
    if i==len(a):
        return
    else:
        r.append(f(a[i]))
        _map(a, f, i+1, r)

def MAP(a, f):
    r = []
    _map(a, f, 0, r)
    return r

def _filter(a, f, i, r):
    if i == len(a):
        return
    else:
        if f(a[i]): r.append(a[i])
        _filter(a, f, i+1, r)

def FILTER(a, f):
    r = []
    _filter(a, f, 0, r)
    return r

def _reduce(a, f, i, r):
    if i == len(a):
        return r
    else:
        r = f(r, a[i])
        return _reduce(a, f, i+1, r)

def REDUCE(a, f, init):
    return _reduce(a, f, 0, init)

if __name__=="__main__":
    a = [1,2,3,4,5]
    EACH(a, lambda x:print(x))
    print(MAP(a, lambda x:x*x))
    print(FILTER(a, lambda x:x%2==1))
    print(REDUCE(a, lambda x,y:x+y, 0))
