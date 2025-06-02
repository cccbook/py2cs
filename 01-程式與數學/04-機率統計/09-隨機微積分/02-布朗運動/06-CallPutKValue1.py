import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# 基本參數
S0 = 100       # 初始價格
T = 1.0        # 到期時間
r = 0.05       # 無風險利率
sigma = 0.2    # 波動率
M = 100000     # 蒙地卡羅模擬次數
np.random.seed(0)

# 履約價範圍
K_list = np.linspace(60, 140, 50)

# 儲存價格
call_bs_list = []
put_bs_list = []
call_mc_list = []
put_mc_list = []

# 蒙地卡羅亂數一次生成（共用）
Z = np.random.randn(M)
S_T_samples = lambda K: S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)

for K in K_list:
    # Black-Scholes 價格
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    call_bs = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    put_bs = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)

    call_bs_list.append(call_bs)
    put_bs_list.append(put_bs)

    # 蒙地卡羅模擬價格
    S_T = S_T_samples(K)
    call_mc = np.exp(-r * T) * np.mean(np.maximum(S_T - K, 0))
    put_mc = np.exp(-r * T) * np.mean(np.maximum(K - S_T, 0))

    call_mc_list.append(call_mc)
    put_mc_list.append(put_mc)

# === 繪圖 ===
plt.figure(figsize=(12, 6))

# Call
plt.subplot(1, 2, 1)
plt.plot(K_list, call_bs_list, label='Call (Black-Scholes)', color='blue')
plt.plot(K_list, call_mc_list, '--', label='Call (Monte Carlo)', color='blue', alpha=0.6)
plt.xlabel('履約價 K')
plt.ylabel('Call 選擇權價格')
plt.title('Call 價格 vs K')
plt.legend()
plt.grid(True)

# Put
plt.subplot(1, 2, 2)
plt.plot(K_list, put_bs_list, label='Put (Black-Scholes)', color='red')
plt.plot(K_list, put_mc_list, '--', label='Put (Monte Carlo)', color='red', alpha=0.6)
plt.xlabel('履約價 K')
plt.ylabel('Put 選擇權價格')
plt.title('Put 價格 vs K')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
