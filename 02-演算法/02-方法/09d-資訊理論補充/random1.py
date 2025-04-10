import random

# 生成 0 到 1 之間的隨機浮點數
random_float = random.random()
print(f"隨機浮點數: {random_float}")

# 生成指定範圍內的隨機整數
random_int = random.randint(1, 10)
print(f"隨機整數: {random_int}")

# 從列表中隨機選擇一個元素
random_choice = random.choice(['蘋果', '香蕉', '橘子'])
print(f"隨機選擇: {random_choice}")

# 洗牌列表中的元素順序
my_list = [1, 2, 3, 4, 5]
random.shuffle(my_list)
print(f"洗牌後的列表: {my_list}")

# 生成指定範圍內的隨機浮點數
random_uniform = random.uniform(5.0, 10.0)
print(f"隨機浮點數（範圍 5.0 到 10.0）: {random_uniform}")
