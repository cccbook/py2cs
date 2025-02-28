from scipy import integrate

# 定義被積函數
def integrand(x, y, z, u, v):
    return x**2 + y**2 + z**2 + u**2 + v**2

# 設定積分範圍
x_lower = 0
x_upper = 1
y_lower = 0
y_upper = 1
z_lower = 0
z_upper = 1
u_lower = 0
u_upper = 1
v_lower = 0
v_upper = 1

# 進行積分
result, error = integrate.nquad(integrand, 
                                 [[x_lower, x_upper], [y_lower, y_upper], 
                                  [z_lower, z_upper], [u_lower, u_upper], 
                                  [v_lower, v_upper]])

print("積分結果:", result)
print("誤差估計:", error)
