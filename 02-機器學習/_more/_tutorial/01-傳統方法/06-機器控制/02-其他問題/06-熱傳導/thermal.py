import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def thermal_system(t, T, C, R, T_ambient, Q):
    dT_dt = (Q(t) - (T - T_ambient) / R) / C
    return dT_dt

C, R = 1.0, 1.0
T_ambient = 25.0  # 環境溫度
Q = lambda t: 10.0  # 恒定加熱功率

y0 = [20.0]  # 初始溫度
t_span = (0, 10)
t_eval = np.linspace(0, 10, 500)
sol = solve_ivp(thermal_system, t_span, y0, args=(C, R, T_ambient, Q), t_eval=t_eval)

plt.plot(sol.t, sol.y[0])
plt.xlabel("Time (s)")
plt.ylabel("Temperature (°C)")
plt.title("Thermal System")
plt.grid()
plt.show()
