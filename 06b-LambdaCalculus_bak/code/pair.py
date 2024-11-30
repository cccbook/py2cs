
"""
function pair(x, y) {
    function dispatch(m) {
        return m === 0 
               ? x
               : m === 1 
               ? y
               : error(m, "argument not 0 or 1 -- pair");
    }
    return dispatch;	      
}
function head(z) { return z(0); }

function tail(z) { return z(1); } 

"""

pair = lambda x:lambda y:lambda sel: x if sel==0 else y 
head = lambda p:p(0)
tail = lambda p:p(1)
car = head
cdr = tail

p = pair(3)(5)
print(f'p(0)={p(0)}')
print(f'p(1)={p(1)}')
print(f'head(p)={head(p)}')
print(f'tail(p)={tail(p)}')
print(f'car(p)={car(p)}')
print(f'cdr(p)={cdr(p)}')

p2 = pair(p)(p)
print(f'p(0)={p(0)}')
