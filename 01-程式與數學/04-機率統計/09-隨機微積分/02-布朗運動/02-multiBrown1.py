import numpy as np
import matplotlib.pyplot as plt

# 模擬參數
T = 1.0           # 總時間
N = 1000          # 時間步數
dt = T / N        # 單步長度
M = 10            # 模擬路徑數量（你可以改成 100 或更多）
t = np.linspace(0, T, N+1)

# GBM 參數
mu = 0.1
sigma = 0.2
S0 = 100

# 模擬 M 條路徑
np.random.seed(42)
dB = np.sqrt(dt) * np.random.randn(M, N)  # 每條路徑的布朗增量
B = np.cumsum(dB, axis=1)                 # 對每條路徑積分得到 B_t
B = np.hstack([np.zeros((M, 1)), B])      # B_0 = 0，加在前面

# 套用 GBM 解析解公式
S = S0 * np.exp((mu - 0.5 * sigma**2) * t + sigma * B)

# 畫圖
plt.figure(figsize=(10, 6))
for i in range(M):
    plt.plot(t, S[i], lw=1)

plt.title("Geometric Brown Motion (GBM) Monte Carlo Simulation")
plt.xlabel("Time t")
plt.ylabel("Asset Value S(t)")
plt.grid(True)
plt.show()
