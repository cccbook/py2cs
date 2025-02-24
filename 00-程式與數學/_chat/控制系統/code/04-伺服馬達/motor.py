import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def dc_motor(t, y, J, b, K_t, K_e, L, R, V):
    theta, omega, i = y
    dtheta_dt = omega
    domega_dt = (K_t * i - b * omega) / J
    di_dt = (V(t) - R * i - K_e * omega) / L
    return [dtheta_dt, domega_dt, di_dt]

J, b, K_t, K_e = 0.01, 0.1, 0.01, 0.01
L, R = 0.5, 1.0
V = lambda t: 5.0  # 恒定電壓 5V

y0 = [0, 0, 0]  # 初始角度, 角速度, 電流
t_span = (0, 10)
t_eval = np.linspace(0, 10, 500)
sol = solve_ivp(dc_motor, t_span, y0, args=(J, b, K_t, K_e, L, R, V), t_eval=t_eval)

plt.plot(sol.t, sol.y[1])
plt.xlabel("Time (s)")
plt.ylabel("Angular Velocity (rad/s)")
plt.title("DC Motor")
plt.grid()
plt.show()
