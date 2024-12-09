import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# 網格參數
nx, ny = 50, 50
dx, dy = 1.0, 1.0
tolerance = 1e-4

# 初始化
u = np.zeros((nx, ny))
u[:, 0] = 1  # 左邊界條件
u[:, -1] = 0  # 右邊界條件
u[0, :] = 0  # 上邊界條件
u[-1, :] = 0  # 下邊界條件

# 數值解：迭代法（高斯-賽德爾）
error = 1.0
while error > tolerance:
    u_new = u.copy()
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            u_new[i, j] = 0.25 * (u[i + 1, j] + u[i - 1, j] + u[i, j + 1] + u[i, j - 1])
    error = np.max(np.abs(u_new - u))
    u = u_new

# 繪圖
plt.imshow(u, extent=(0, nx * dx, 0, ny * dy), origin='lower', cmap='viridis')
plt.colorbar(label="u(x, y)")
plt.title("拉普拉斯方程數值解")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
