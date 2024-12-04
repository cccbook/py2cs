PAIR = lambda x:lambda y:lambda sel: x if sel==0 else y 
HEAD = lambda p:p(0)
TAIL = lambda p:p(1)
CONS = PAIR
CAR = HEAD
CDR = TAIL

p = PAIR(3)(5)
print(f'p(0)={p(0)}')
print(f'p(1)={p(1)}')
print(f'HEAD(p)={HEAD(p)}')
print(f'TAIL(p)={TAIL(p)}')
print(f'CAR(p)={CAR(p)}')
print(f'CDR(p)={CDR(p)}')

p2 = PAIR(p)(p)
print(f'HEAD(HEAD(p2))={HEAD(HEAD(p2))}')
print(f'TAIL(HEAD(p2))={TAIL(HEAD(p2))}')

RANGE = lambda m:lambda n:PAIR(m)(None) if m==n else PAIR(m)(RANGE(m+1)(n))

print('-------------RANGE----------')

r=RANGE(3)(5)
print(f'r=RANGE(3)(5)')
print(f'HEAD(r)={HEAD(r)}')
print(f'TAIL(r)={TAIL(r)}')
print(f'HEAD(TAIL(r))={HEAD(TAIL(r))}')
print(f'TAIL(TAIL(r))={TAIL(TAIL(r))}')
print(f'HEAD(TAIL(TAIL(r)))={HEAD(TAIL(TAIL(r)))}')
print(f'TAIL(TAIL(TAIL(r)))={TAIL(TAIL(TAIL(r)))}')

print('-------------EACH----------')
EACH = lambda x:lambda f:\
  f(x) if x == None or isinstance(x,int)\
  else (f(HEAD(x)),EACH(TAIL(x))(f))[-1]

EACH(r)(lambda x:print(x))

print('-------------MAP----------')

MAP = lambda x:lambda f: \
  None if x==None \
  else f(x) if isinstance(x,int) \
  else PAIR(MAP(HEAD(x))(f))(MAP(TAIL(x))(f))

m = MAP(r)(lambda x:x*2)

EACH(m)(print)
