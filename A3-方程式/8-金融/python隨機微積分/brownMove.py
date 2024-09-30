import numpy as np
import matplotlib.pyplot as plt

# 模擬參數
T = 1.0       # 總時間
N = 500       # 時間步數
dt = T / N    # 每步的時間
mu = 0.0      # 漂移率
sigma = 1.0   # 波動率

# 模擬布朗運動
def simulate_brownian_motion(T, N, mu, sigma):
    dt = T / N
    # 隨機增量
    dB = np.random.normal(0, np.sqrt(dt), N)
    # 布朗運動路徑
    B = np.zeros(N)
    for i in range(1, N):
        B[i] = B[i-1] + mu * dt + sigma * dB[i-1]
    return B

# 生成布朗運動
brownian_motion = simulate_brownian_motion(T, N, mu, sigma)

# 可視化布朗運動
plt.figure(figsize=(10, 5))
plt.plot(np.linspace(0, T, N), brownian_motion, label='Brownian Motion')
plt.title('Simulation of Brownian Motion')
plt.xlabel('Time')
plt.ylabel('B(t)')
plt.legend()
plt.grid()
plt.show()
