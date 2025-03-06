from pair import *

'''
def pair(x, y):
    return lambda m : m(x, y)

def head(z):
    return z(lambda p,q : p)

def tail(z):
    return z(lambda p,q : q)
'''

a = pair(1, 2)
print('head(a)=', head(a))
print('tail(a)=', tail(a))
# tail(a) = tail(pair(1,2)) = pair(1,2)(lambda p,q:q)
# = (lambda m : m(1, 2))(lambda p,q:q) 注意：m=(lambda p,q:q)
# = (lambda p,q:q)(1,2) = 2
b = pair(3, 4)
c = pair(a, b)

print('head(head(c))=', head(head(c)))
print('head(tail(c))=', head(tail(c)))
