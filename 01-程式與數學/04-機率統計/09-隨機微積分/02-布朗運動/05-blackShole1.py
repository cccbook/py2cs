import numpy as np
from scipy.stats import norm

# 參數
S0 = 100     # 初始價格
K = 110      # 履約價
T = 1.0      # 到期時間
r = 0.05     # 無風險利率
sigma = 0.2  # 波動率
M = 100000   # 蒙地卡羅模擬數量

# === Black-Scholes 理論價格 ===
d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
d2 = d1 - sigma * np.sqrt(T)

call_bs = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
put_bs = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)

# === 蒙地卡羅模擬價格 ===
np.random.seed(42)
Z = np.random.randn(M)
S_T = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)

call_mc = np.exp(-r * T) * np.mean(np.maximum(S_T - K, 0))
put_mc  = np.exp(-r * T) * np.mean(np.maximum(K - S_T, 0))

# === 輸出對照 ===
print("歐式選擇權價格對照：")
print(f"{'選擇權':<10}{'蒙地卡羅':>12}{'Black-Scholes':>18}")
print(f"{'Call':<10}{call_mc:12.4f}{call_bs:18.4f}")
print(f"{'Put':<10}{put_mc:12.4f}{put_bs:18.4f}")
