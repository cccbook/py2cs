import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def lorenz(t, state, sigma, rho, beta):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

# 初始條件和參數
sigma, rho, beta = 10.0, 28.0, 8.0/3.0
state0 = [1.0, 1.0, 1.0]
t_span = (0, 50)
t_eval = np.linspace(t_span[0], t_span[1], 10000)

# 數值解
solution = solve_ivp(lorenz, t_span, state0, args=(sigma, rho, beta), t_eval=t_eval)

# 繪圖
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot(solution.y[0], solution.y[1], solution.y[2])
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_title("洛倫茲吸引子")
plt.show()
