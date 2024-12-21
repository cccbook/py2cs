import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# 參數
L = 10
T = 2
dx = 0.1
dt = 0.01
nx = int(L / dx) + 1
nt = int(T / dt) + 1

x = np.linspace(0, L, nx)
u = np.zeros(nx)
u[int(nx / 4):int(nx / 2)] = 1  # 初始條件：方波

# 數值解：Lax-Friedrichs 方法
for n in range(nt):
    u_new = np.zeros(nx)
    for i in range(1, nx - 1):
        u_new[i] = 0.5 * (u[i + 1] + u[i - 1]) - dt / (2 * dx) * u[i] * (u[i + 1] - u[i - 1])
    u = u_new

# 繪圖
plt.plot(x, u, label="速度場 u(x)")
plt.xlabel("x")
plt.ylabel("u")
plt.title("簡化納維-斯托克斯數值解")
plt.legend()
plt.show()
