# Python 生成器的例子
def lazy_range(n):
    i = 0
    while i < n:
        yield i
        i += 1

# 使用生成器
for num in lazy_range(5):
    print(num)
