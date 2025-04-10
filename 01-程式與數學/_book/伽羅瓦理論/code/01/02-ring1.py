# 檢查環的基本性質：整數模 6 的加法和乘法環
class Ring:
    def __init__(self, elements, add_op, mul_op):
        self.elements = elements
        self.add_op = add_op
        self.mul_op = mul_op
    
    def check_add_group(self):
        # 檢查加法群性質（結合性、單位元、逆元）
        for a in self.elements:
            for b in self.elements:
                for c in self.elements:
                    if self.add_op(self.add_op(a, b), c) != self.add_op(a, self.add_op(b, c)):
                        return False
        return True
    
    def check_distributive_law(self):
        # 檢查分配律：a * (b + c) == a * b + a * c
        for a in self.elements:
            for b in self.elements:
                for c in self.elements:
                    if self.mul_op(a, self.add_op(b, c)) != self.add_op(self.mul_op(a, b), self.mul_op(a, c)):
                        return False
        return True
    
    def add_op(self, a, b):
        return (a + b) % 6

    def mul_op(self, a, b):
        return (a * b) % 6

# 定義模 6 的加法與乘法環
Z_6_ring = Ring([0, 1, 2, 3, 4, 5], lambda a, b: (a + b) % 6, lambda a, b: (a * b) % 6)

# 檢查加法群性質
print("Is add group satisfied?", Z_6_ring.check_add_group())

# 檢查分配律
print("Is distributive law satisfied?", Z_6_ring.check_distributive_law())
