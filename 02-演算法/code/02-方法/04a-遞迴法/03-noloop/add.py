def inc(n):return n+1
def dec(n):return n-1

def add(a, b):
    return b if a == 0 else add(dec(a), inc(b))

print('add(3,2)=', add(3,2))
