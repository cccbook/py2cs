# 定義多項式 x^2 - 2x + 1（不可分多項式）
polynomial_inseparable = x**2 - 2*x + 1

# 計算多項式的導數
derivative = sp.diff(polynomial_inseparable, x)

# 檢查多項式與導數的最大公因數（GCD）
gcd = sp.gcd(polynomial_inseparable, derivative)

# 判斷多項式是否可分
if gcd == 1:
    print("The polynomial is separable.")
else:
    print("The polynomial is inseparable.")
