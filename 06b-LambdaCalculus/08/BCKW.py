B = lambda f:lambda g: lambda x: f(g(x)) # Compose = f*g
C = lambda f:lambda g: lambda x: g(f(x)) # 交換
W = lambda f:lambda x: f(f(x)) # 重複
K = lambda x:lambda y: x # 丟棄 y

Y = lambda f:\
  (lambda x:f(lambda y:x(x)(y)))\
  (lambda x:f(lambda y:x(x)(y)))

Y2 = lambda f:W(B f(B W(W K K)))

# 階層 FACTORIAL(n) = n!
FACTORIAL = Y(lambda f:lambda n:\
  1 if n==0\
  else n*f(n-1)\
)

print(f'FACTORIAL(3)={FACTORIAL(3)}')
