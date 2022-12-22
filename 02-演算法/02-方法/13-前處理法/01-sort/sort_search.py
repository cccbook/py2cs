data = ["c", "a", "b", "e", "g", "d"]

data.sort()

print('data=', data)

def _bsearch(a, x, low, high):
    mid = (low+high)//2
    if a[mid] == x:
        return mid
    elif a[mid]<x:
        return _bsearch(a, x, mid+1, high)
    else:
        return _bsearch(a, x, low, mid-1)

def bsearch(a, x):
    return _bsearch(a, x, 0, len(a))

print('bsearch(data, "c")=', bsearch(data, "c"))
