import fp

def set(r, i):
    r['i'] = i

def linearSearch(a, x):
    n = len(a)
    r = { 'i': -1 }
    fp.each(range(0,n), lambda i:
        set(r, i) if x==a[i] else None
    )
    return r['i']

a = [3,7,2,6,8,4]
print('a=', a)
print('linearSearch(a, 6)=', linearSearch(a, 6))

