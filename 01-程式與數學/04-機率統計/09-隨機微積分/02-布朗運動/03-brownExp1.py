import numpy as np
import matplotlib.pyplot as plt

# 參數設定
T = 1.0           # 總時間
N = 1000          # 時間步數
dt = T / N
M = 10000         # 模擬路徑數量（越大越準確）
mu = 0.1          # 漂移率
sigma = 0.2       # 波動率
S0 = 100          # 初始價格

# 模擬布朗運動與 GBM
np.random.seed(123)
dB = np.sqrt(dt) * np.random.randn(M, N)
B_T = np.sum(dB, axis=1)  # 每條路徑的 B_T
S_T = S0 * np.exp((mu - 0.5 * sigma**2) * T + sigma * B_T)

# 理論期望
E_theoretical = S0 * np.exp(mu * T)

# 統計分析
E_empirical = np.mean(S_T)
std_empirical = np.std(S_T)

# 顯示結果
print(f"模擬期望值： {E_empirical:.4f}")
print(f"理論期望值： {E_theoretical:.4f}")
print(f"標準差：     {std_empirical:.4f}")

# 繪製直方圖
plt.hist(S_T, bins=50, density=True, alpha=0.7, color='skyblue')
plt.axvline(E_empirical, color='red', linestyle='--', label='Expectation of Simulation')
plt.axvline(E_theoretical, color='green', linestyle='--', label='Expectation of Theoretical')
plt.title("GBM Distributation of S(T) Simulation")
plt.xlabel("S(T)")
plt.ylabel("Probability Density")
plt.legend()
plt.grid(True)
plt.show()
