# 如果 Fibonacci 用 If 而非 If_lazy, 
# 那麼 Fibonacci(n-1) 會馬上運算
# 結果就是 8=>7=>....0=>-1=>-2=>.... 當掉

If = lambda cond, a, b: a() if cond else b()
# 採用 If_lazy, 並要求 a, b 必須是函數，這樣就可以阻止 a, b 立即算出

Fibonacci = lambda n: \
  If(n<2, lambda:n, lambda:Fibonacci(n-1)+Fibonacci(n-2))

print('Fibonacci(8)=', Fibonacci(8))

