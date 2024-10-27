def count_up_to(max):
    count = 1
    while count <= max:
        yield count  # 返回當前計數並暫停函數
        count += 1   # 增加計數

# 使用生成器
counter = count_up_to(5)

for number in counter:
    print(number)
