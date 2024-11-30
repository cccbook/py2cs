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

print(f'CAR(r)={CAR(r)}')
print(f'CDR(r)={CDR(r)}')
print(f'CAR(CDR(r))={CAR(CDR(r))}')
print(f'CDR(CDR(r))={CDR(CDR(r))}')
print(f'CAR(CDR(CDR(r)))={CAR(CDR(CDR(r)))}')
print(f'CDR(CDR(CDR(r)))={CDR(CDR(CDR(r)))}')
