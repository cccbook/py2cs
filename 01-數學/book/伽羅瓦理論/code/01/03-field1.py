# 檢查域的基本性質：有限域 F_5
class Field(Ring):
    def __init__(self, elements, add_op, mul_op):
        super().__init__(elements, add_op, mul_op)
    
    def check_multiplicative_group(self):
        # 檢查乘法群性質（除 0 外所有元素都有逆元）
        for a in self.elements:
            if a != 0:
                has_inverse = False
                for b in self.elements:
                    if self.mul_op(a, b) == 1:
                        has_inverse = True
                        break
                if not has_inverse:
                    return False
        return True

# 定義模 5 的有限域 F_5
F_5 = Field([0, 1, 2, 3, 4], lambda a, b: (a + b) % 5, lambda a, b: (a * b) % 5)

# 檢查加法群性質
print("Is add group satisfied?", F_5.check_add_group())

# 檢查分配律
print("Is distributive law satisfied?", F_5.check_distributive_law())

# 檢查乘法群性質
print("Is multiplicative group satisfied?", F_5.check_multiplicative_group())
