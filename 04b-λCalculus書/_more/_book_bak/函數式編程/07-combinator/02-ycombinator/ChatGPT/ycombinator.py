# 定義 Y-Combinator
def Y(f):
    return (lambda x: f(lambda v: x(x)(v)))(lambda x: f(lambda v: x(x)(v)))

# 使用 Y-Combinator 實現階乘函數
factorial = Y(lambda f: lambda n: 1 if n == 0 else n * f(n - 1))

# 測試
print(factorial(5))  # 輸出 120
