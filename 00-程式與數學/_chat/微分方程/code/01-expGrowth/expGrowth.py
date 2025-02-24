import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# 定義微分方程
def exponential_growth(t, y, k):
    return k * y

# 初始條件和參數
y0 = 1.0
k = 0.5
t_span = (0, 10)  # 時間範圍
t_eval = np.linspace(t_span[0], t_span[1], 100)

# 數值解
solution = solve_ivp(exponential_growth, t_span, [y0], args=(k,), t_eval=t_eval)

# 繪圖
plt.plot(solution.t, solution.y[0], label="數值解")
plt.xlabel("時間 t")
plt.ylabel("y(t)")
plt.title("指數增長與衰減")
plt.legend()
plt.show()
