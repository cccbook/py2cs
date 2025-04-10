def binArraySearch(a, o, ifrom, ito):
    if ito < ifrom: return -1
    mid = (ito+ifrom)//2
    if a[mid]==o: return mid
    elif o>a[mid]: return binArraySearch(a, o, mid+1, ito)
    else: return binArraySearch(a, o, ifrom, mid-1)

def binSearch(a, o):
    return binArraySearch(a, o, 0, len(a))

a = [1, 4, 7, 9, 13, 19]
o = 13
print(binSearch(a, o))
o = 10
print(binSearch(a, o))
