If  = lambda c:lambda a:lambda b:a if c else b
Max = lambda a: lambda b: If(a>b)(a)(b)
Min = lambda a: lambda b: If(a<b)(a)(b)

# 如果 Fibonacci 用 If 而非 If_n, 那麼 Fibonacci(n-1) 會馬上運算，結果就是 8=>7=>....0=>-1=>-2=>.... 當掉
If_lazy = lambda cond: lambda a: lambda b: a() if cond else b()
Fibonacci = lambda n: If_lazy(n<2)(lambda:n)(lambda:Fibonacci(n-1)+Fibonacci(n-2))

print('Max(3)(5)=', Max(3)(5))
print('Min(3)(5)=', Min(3)(5))
print('Fibonacci(8)=', Fibonacci(8))

