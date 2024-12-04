If  = lambda c:lambda a:lambda b:a if c else b
Max = lambda a: lambda b: If(a>b)(a)(b)
Min = lambda a: lambda b: If(a<b)(a)(b)
If_lazy = lambda cond: lambda a: lambda b: a() if cond else b()
Fibonacci = lambda n: If_lazy(n<2)(lambda:n)(lambda:Fibonacci(n-1)+Fibonacci(n-2))
n = 5
print('n=', n, 'If(n>3)("n>3")("n<=3")=', If(n>3)("n>3")("n<=3"))
n = 1
print('n=', n, 'If(n>3)("n>3")("n<=3")=', If(n>3)("n>3")("n<=3"))
print('Max(3)(5)=', Max(3)(5))
print('Min(3)(5)=', Min(3)(5))
print('Fibonacci(8)=', Fibonacci(8))

# Combinators : 把傳進來的函數 f 重複兩次，變成 lambda f:f(f)
# 例如 print(Y(lambda f: lambda n: IF(n < 2)(lambda: n)(lambda: f(n-1) + f(n-2))())(8))
Y = lambda f:\
  (lambda x:f(lambda y:x(x)(y)))\
  (lambda x:f(lambda y:x(x)(y)))

# Y(fib)= fib(x)=>fib(y=>x(x)(y)) fib(x)=>fib(y=>x(x)(y))
# Y(fib)(n) = fib(fib)(n) ???
print('Y(f)(8)=', Y(lambda f: \
  lambda n: If(n < 2)(lambda: n)(lambda: f(n-1) + f(n-2))())(8))
