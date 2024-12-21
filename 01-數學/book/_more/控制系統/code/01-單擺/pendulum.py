import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def simple_pendulum(t, y, g, L):
    theta, omega = y
    dydt = [omega, -g / L * np.sin(theta)]
    return dydt

g = 9.81  # 重力加速度
L = 1.0   # 擺長
y0 = [np.pi / 4, 0]  # 初始條件：擺角 45 度，初始角速度 0
t_span = (0, 10)  # 時間範圍
t_eval = np.linspace(*t_span, 500)

sol = solve_ivp(simple_pendulum, t_span, y0, args=(g, L), t_eval=t_eval)

plt.plot(sol.t, sol.y[0])
plt.xlabel("Time (s)")
plt.ylabel("Angle (rad)")
plt.title("Simple Pendulum")
plt.grid()
plt.show()
