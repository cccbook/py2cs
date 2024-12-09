import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def rlc_circuit(t, y, L, R, C, V):
    q, i = y
    dydt = [i, (V(t) - R * i - q / C) / L]
    return dydt

L, R, C = 1.0, 1.0, 1.0  # 電感, 電阻, 電容
V = lambda t: np.sin(t)  # 電壓源

y0 = [0, 0]  # 初始電荷和電流
t_span = (0, 30)
t_eval = np.linspace(0, 30, 500)
sol = solve_ivp(rlc_circuit, t_span, y0, args=(L, R, C, V), t_eval=t_eval)

plt.plot(sol.t, sol.y[1])
plt.xlabel("Time (s)")
plt.ylabel("Current (A)")
plt.title("RLC Circuit")
plt.grid()
plt.show()
