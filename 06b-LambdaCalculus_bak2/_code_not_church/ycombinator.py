Y = lambda f:\
  (lambda x:f(lambda y:x(x)(y)))\
  (lambda x:f(lambda y:x(x)(y)))

# 階層 FACTORIAL(n) = n!
FACTORIAL = Y(lambda f:lambda n:\
  1 if n==0\
  else n*f(n-1)\
)

print(f'FACTORIAL(3)={FACTORIAL(3)}')
print(f'FACTORIAL(5)={FACTORIAL(5)}')
