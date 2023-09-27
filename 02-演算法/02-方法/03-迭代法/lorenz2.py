import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# 定义 Lorenz 吸引子的微分方程
def lorenz(t, y, sigma, rho, beta):
    dx_dt = sigma * (y[1] - y[0])
    dy_dt = y[0] * (rho - y[2]) - y[1]
    dz_dt = y[0] * y[1] - beta * y[2]
    return [dx_dt, dy_dt, dz_dt]

# 设置参数
sigma = 10.0
rho = 28.0
beta = 8.0 / 3.0

# 初始条件
y0 = [1.0, 0.0, 0.0]

# 定义时间间隔
t_span = (0, 100) # t_span = (0, 25)

t_eval = np.linspace(*t_span, 10000)

# 解微分方程
sol = solve_ivp(lorenz, t_span, y0, args=(sigma, rho, beta), t_eval=t_eval)

# 绘制 Lorenz attractor
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(sol.y[0], sol.y[1], sol.y[2], lw=0.5)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Lorenz Attractor')
plt.show()
