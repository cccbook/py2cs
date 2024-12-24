import numpy as np

# 定義多項式類
class Polynomial:
    def __init__(self, coefficients):
        # coefficients 是多項式的係數，從最高次項開始
        self.coefficients = np.array(coefficients)

    def evaluate(self, x):
        # 計算多項式的值
        return sum(c * x**i for i, c in enumerate(reversed(self.coefficients)))
    
    def degree(self):
        # 返回多項式的次數
        return len(self.coefficients) - 1
    
    def __add__(self, other):
        # 多項式加法
        degree = max(self.degree(), other.degree())
        new_coeffs = [0] * (degree + 1)
        
        for i in range(self.degree() + 1):
            new_coeffs[degree - i] += self.coefficients[-(i + 1)]
        
        for i in range(other.degree() + 1):
            new_coeffs[degree - i] += other.coefficients[-(i + 1)]
        
        return Polynomial(new_coeffs)

    def __mul__(self, other):
        # 多項式乘法
        degree = self.degree() + other.degree()
        new_coeffs = [0] * (degree + 1)
        
        for i, a in enumerate(reversed(self.coefficients)):
            for j, b in enumerate(reversed(other.coefficients)):
                new_coeffs[degree - (i + j)] += a * b
        
        return Polynomial(new_coeffs)

# 測試多項式的基本操作
P = Polynomial([1, -3, 2])  # P(x) = x^2 - 3x + 2
Q = Polynomial([2, 1])       # Q(x) = 2x + 1

print("P(x) =", P.coefficients)
print("Q(x) =", Q.coefficients)

# 計算 P(3) 和 Q(3)
print("P(3) =", P.evaluate(3))
print("Q(3) =", Q.evaluate(3))

# 多項式加法
R = P + Q
print("P(x) + Q(x) =", R.coefficients)

# 多項式乘法
S = P * Q
print("P(x) * Q(x) =", S.coefficients)
