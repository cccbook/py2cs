# 建立一個名為 "table" 的字典，這個字典就是我們的查表表格
table = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}

# 接下來，我們就可以透過 key 值 (例如 "A"、"B"、"C" 等) 來查詢 value 值
print(table["A"])  # 輸出 1
print(table["B"])  # 輸出 2

# 我們也可以利用迴圈來輸出所有 key-value 對
for key, value in table.items():
  print(key, value)

# 輸出結果：
# A 1
# B 2
# C 3
# D 4
# E 5
