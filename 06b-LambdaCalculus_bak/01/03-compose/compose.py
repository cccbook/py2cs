# 函數組合
def compose(f, g):
    return lambda x: f(g(x))

# 定義兩個簡單的函數
add2 = lambda x: x + 2
multiply_by_3 = lambda x: x * 3

# 使用函數組合
add_then_multiply = compose(multiply_by_3, add2)

print(add_then_multiply(5))  # 輸出 21 (即 (5 + 2) * 3)