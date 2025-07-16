import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# 基本參數
S0 = 100      # 初始資產價格
K = 100       # 履約價格
T = 1.0       # 到期時間
r = 0.05      # 無風險利率
sigma = 0.2   # 波動率

# 計算 Black-Scholes 選擇權價格
d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
d2 = d1 - sigma * np.sqrt(T)

call_price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
put_price  = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)

# 模擬不同到期價格 S_T
S_T = np.linspace(50, 150, 200)

# 損益計算
call_payoff = np.maximum(S_T - K, 0) - call_price
put_payoff = np.maximum(K - S_T, 0) - put_price

# 繪圖
plt.figure(figsize=(10, 6))
plt.axhline(0, color='black', linestyle='--')

plt.plot(S_T, call_payoff, label=f'Call Payoff (K={K})', color='blue')
plt.plot(S_T, put_payoff, label=f'Put Payoff (K={K})', color='red')

plt.title('歐式選擇權的損益圖（買方）')
plt.xlabel('到期資產價格 $S_T$')
plt.ylabel('損益 (Payoff)')
plt.legend()
plt.grid(True)
plt.show()
