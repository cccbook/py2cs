import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def mass_spring_damper(t, y, m, c, k, F):
    x, v = y
    dydt = [v, (F(t) - c * v - k * x) / m]
    return dydt

m, c, k = 1.0, 0.5, 2.0  # 質量, 阻尼, 彈簧常數
F = lambda t: np.sin(t)  # 外力

y0 = [0, 0]  # 初始位移和速度
t_span = (0, 10)
sol = solve_ivp(mass_spring_damper, t_span, y0, args=(m, c, k, F), t_eval=t_eval)

plt.plot(sol.t, sol.y[0])
plt.xlabel("Time (s)")
plt.ylabel("Displacement (m)")
plt.title("Mass-Spring-Damper System")
plt.grid()
plt.show()
