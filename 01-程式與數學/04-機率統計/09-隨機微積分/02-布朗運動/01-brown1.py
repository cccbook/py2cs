import numpy as np
import matplotlib.pyplot as plt

# 參數設定
T = 1.0         # 總時間
N = 1000        # 時間分割數
dt = T / N      # 每步時間
t = np.linspace(0, T, N+1)

mu = 0.1        # 漂移率
sigma = 0.2     # 波動率
S0 = 100        # 初始價格

# 模擬布朗運動
np.random.seed(1)
dB = np.sqrt(dt) * np.random.randn(N)
B = np.concatenate(([0], np.cumsum(dB)))

# Euler-Maruyama 方法模擬 GBM
S = np.zeros(N+1)
S[0] = S0
for i in range(N):
    S[i+1] = S[i] + mu * S[i] * dt + sigma * S[i] * dB[i]

# 理論解
S_exact = S0 * np.exp((mu - 0.5 * sigma**2) * t + sigma * B)

# 畫圖
plt.plot(t, S, label='Euler Simulation Solution')
plt.plot(t, S_exact, '--', label='Theoretic Solution', alpha=0.8)
plt.title("Geometric Brown Mothon (GBM) Simulation")
plt.xlabel("Time t")
plt.ylabel("Asset Value S(t)")
plt.legend()
plt.grid(True)
plt.show()
