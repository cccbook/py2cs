# 檢查群的基本性質：加法群 Z_6
class Group:
    def __init__(self, elements, operation):
        self.elements = elements
        self.operation = operation
        self.identity = None
        self.inverses = {}
        
    def check_associativity(self):
        for a in self.elements:
            for b in self.elements:
                for c in self.elements:
                    if self.operation(self.operation(a, b), c) != self.operation(a, self.operation(b, c)):
                        return False
        return True
    
    def find_identity(self):
        for e in self.elements:
            if all(self.operation(e, x) == x and self.operation(x, e) == x for x in self.elements):
                self.identity = e
                return e
        return None
    
    def find_inverses(self):
        for x in self.elements:
            for y in self.elements:
                if self.operation(x, y) == self.identity:
                    self.inverses[x] = y
                    break
    
    def operation(self, a, b):
        return (a + b) % 6

# 定義 Z_6 群
Z_6 = Group([0, 1, 2, 3, 4, 5], lambda a, b: (a + b) % 6)

# 檢查結合性
print("Is associativity satisfied?", Z_6.check_associativity())

# 找到單位元
identity = Z_6.find_identity()
print("Identity element:", identity)

# 找到逆元
Z_6.find_inverses()
print("Inverses:", Z_6.inverses)
