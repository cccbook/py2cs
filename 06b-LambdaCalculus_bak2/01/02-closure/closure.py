# 閉包的例子
def make_multiplier(factor):
    return lambda x: x * factor

# 使用閉包
multiply_by_2 = make_multiplier(2)
multiply_by_3 = make_multiplier(3)

print(multiply_by_2(5))  # 輸出 10
print(multiply_by_3(5))  # 輸出 15