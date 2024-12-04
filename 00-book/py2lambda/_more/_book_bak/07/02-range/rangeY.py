# 定義布林值
TRUE = lambda x: lambda y: x
FALSE = lambda x: lambda y: y

# Lists
CONS = lambda x:lambda y:lambda f:f(x)(y)
CAR  = lambda p:p(TRUE)
CDR  = lambda p:p(FALSE)

Y = lambda f:\
  (lambda x:f(lambda y:x(x)(y)))\
  (lambda x:f(lambda y:x(x)(y)))

RANGE = lambda m:lambda n:Y(lambda f:lambda m:\
  (lambda _: CONS(m)(None)) if m==n\
  else (lambda _: CONS(m)(f(m+1)))\
(None))(m)

r=RANGE(3)(5)

print(f'CAR(r)={CAR(r)}')
print(f'CAR(CDR(r))={CAR(CDR(r))}')