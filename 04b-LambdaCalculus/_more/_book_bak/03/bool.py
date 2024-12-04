# 定義布林值
TRUE = lambda x: lambda y: x
FALSE = lambda x: lambda y: y

# 定義 IF 條件
IF = lambda c: lambda x: lambda y: c(x)(y)

# 測試
print(IF(TRUE)('Yes')('No'))  # 輸出 'Yes'
print(IF(FALSE)('Yes')('No')) # 輸出 'No'

# 定義 NOT
NOT = lambda c: c(FALSE)(TRUE)

# 測試 NOT
assert NOT(TRUE) == FALSE
assert NOT(FALSE) == TRUE

# 定義 AND
AND = lambda p: lambda q: p(q)(FALSE)

# 測試 AND
assert AND(TRUE)(TRUE) == TRUE
assert AND(TRUE)(FALSE) == FALSE
assert AND(FALSE)(TRUE) == FALSE
assert AND(FALSE)(FALSE) == FALSE

# 定義 OR
OR = lambda p: lambda q: p(TRUE)(q)

# 測試 OR
assert OR(TRUE)(TRUE) == TRUE
assert OR(TRUE)(FALSE) == TRUE
assert OR(FALSE)(TRUE) == TRUE
assert OR(FALSE)(FALSE) == FALSE

# 定義 XOR
XOR = lambda p: lambda q: p(NOT(q))(q)

# 測試 XOR
assert XOR(TRUE)(TRUE) == FALSE
assert XOR(TRUE)(FALSE) == TRUE
assert XOR(FALSE)(TRUE) == TRUE
assert XOR(FALSE)(FALSE) == FALSE

# 定義 NOR
NOR = lambda p: lambda q: NOT(OR(p)(q))

# 測試 NOR
assert NOR(TRUE)(TRUE) == FALSE
assert NOR(TRUE)(FALSE) == FALSE
assert NOR(FALSE)(TRUE) == FALSE
assert NOR(FALSE)(FALSE) == TRUE

# 定義 NAND
NAND = lambda p: lambda q: NOT(AND(p)(q))

# 測試 NAND
assert NAND(TRUE)(TRUE) == FALSE
assert NAND(TRUE)(FALSE) == TRUE
assert NAND(FALSE)(TRUE) == TRUE
assert NAND(FALSE)(FALSE) == TRUE

# 透過邏輯實現條件選擇
CHOICE = lambda p: IF(p)('Option A')('Option B')

# 測試
assert CHOICE(TRUE) == 'Option A'
assert CHOICE(FALSE) == 'Option B'

# 定義條件: (TRUE AND FALSE) OR TRUE
COMPLEX_CONDITION = OR(AND(TRUE)(FALSE))(TRUE)

# 測試條件
assert IF(COMPLEX_CONDITION)('Pass')('Fail') == 'Pass'
