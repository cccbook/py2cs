import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def inverted_pendulum(t, y, M, m, l, g, F):
    x, v, theta, omega = y
    sin_theta, cos_theta = np.sin(theta), np.cos(theta)
    denom = M + m * (1 - cos_theta**2)

    dx_dt = v
    dv_dt = (F(t) - m * l * omega**2 * sin_theta + m * g * sin_theta * cos_theta) / denom
    dtheta_dt = omega
    domega_dt = (-g * sin_theta - cos_theta * dv_dt) / l

    return [dx_dt, dv_dt, dtheta_dt, domega_dt]

M, m, l, g = 1.0, 0.1, 1.0, 9.81
F = lambda t: 0.0  # 無外力

y0 = [0, 0, np.pi / 8, 0]  # 初始位置, 速度, 角度, 角速度
t_span = (0, 10)
t_eval = np.linspace(0, 10, 500)
sol = solve_ivp(inverted_pendulum, t_span, y0, args=(M, m, l, g, F), t_eval=t_eval)

plt.plot(sol.t, sol.y[2])
plt.xlabel("Time (s)")
plt.ylabel("Pendulum Angle (rad)")
plt.title("Inverted Pendulum")
plt.grid()
plt.show()
