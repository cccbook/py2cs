def pair(x, y):
    return lambda m : m(x, y)

def head(z):
    return z(lambda p,q : p)

def tail(z):
    return z(lambda p,q : q)
