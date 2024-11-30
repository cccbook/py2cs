# 定義布林值
TRUE = lambda x: lambda y: x
FALSE = lambda x: lambda y: y

# Lists
CONS = lambda x:lambda y:lambda f:f(x)(y)
CAR  = lambda p:p(TRUE)
CDR  = lambda p:p(FALSE)

RANGE = lambda m:lambda n:\
  CONS(m)(None) if m==n\
  else CONS(m)(RANGE(m+1)(n))

r=RANGE(3)(5)

MAP = lambda x:lambda f:\
  None if x==None\
  else f(x) if isinstance(x, int)\
  else CONS(MAP(CAR(x)))(MAP(CDR(x)))

m=MAP(r)(lambda x:2*x)
print(f'CAR(m)={CAR(m)}')
print(f'CAR(CDR(m))={CAR(CDR(m))}')

t=MAP(5)(lambda x:2*x)
print('t=',t)

p=CONS(3)(5)
p2=MAP(p)(lambda x:2*x)
print(f'CAR(p)={CAR(p)}')
print(f'CAR(p2)={CAR(p2)}')

