import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def simple_harmonic(t, y, omega):
    x, v = y  # y[0] = x, y[1] = v = dx/dt
    return [v, -omega**2 * x]

# 初始條件和參數
x0, v0 = 1.0, 0.0  # 初始位移和速度
omega = 2.0
y0 = [x0, v0]
t_span = (0, 10)
t_eval = np.linspace(t_span[0], t_span[1], 100)

# 數值解
solution = solve_ivp(simple_harmonic, t_span, y0, args=(omega,), t_eval=t_eval)

# 繪圖
plt.plot(solution.t, solution.y[0], label="位移 x(t)")
plt.plot(solution.t, solution.y[1], label="速度 v(t)")
plt.xlabel("時間 t")
plt.ylabel("位移與速度")
plt.title("簡諧運動")
plt.legend()
plt.show()
