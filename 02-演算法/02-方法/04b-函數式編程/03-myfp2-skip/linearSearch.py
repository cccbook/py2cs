import fp

ri = -1

def setRi(value):
    global ri
    ri = value

def linearSearch(a, x):
    global ri
    n = len(a)
    ri = -1
    fp.each(range(0,n), lambda i:
        setRi(i) if x==a[i] else None
    )
    return ri

a = [3,7,2,6,8,4]
print('a=', a)
print('linearSearch(a, 6)=', linearSearch(a, 6))

