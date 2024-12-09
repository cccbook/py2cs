import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# 參數
L = 10  # 長度
T = 5   # 時間
c = 1.0  # 波速
dx = 0.1
dt = 0.01
nx = int(L / dx) + 1
nt = int(T / dt) + 1

# 初始條件
x = np.linspace(0, L, nx)
u = np.zeros((nt, nx))
u[0, :] = np.sin(np.pi * x / L)  # 初始位形
u[:, 0] = 0  # 邊界條件
u[:, -1] = 0

# 數值解：有限差分
for n in range(0, nt - 1):
    for i in range(1, nx - 1):
        u[n + 1, i] = 2 * (1 - c**2 * dt**2 / dx**2) * u[n, i] - u[n - 1, i] + \
                      c**2 * dt**2 / dx**2 * (u[n, i + 1] + u[n, i - 1])

# 繪圖
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots()
line, = ax.plot(x, u[0, :])

def update(frame):
    line.set_ydata(u[frame, :])
    return line,

ani = FuncAnimation(fig, update, frames=range(0, nt, 5), interval=50)
plt.show()
