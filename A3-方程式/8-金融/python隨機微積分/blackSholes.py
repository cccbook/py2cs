import numpy as np
import matplotlib.pyplot as plt

# 參數設定
S0 = 100.0   # 初始股票價格
K = 100.0    # 行使價格
r = 0.05     # 無風險利率
sigma = 0.2  # 波動率
T = 1.0      # 到期時間
N = 1000     # 模擬步數
dt = T / N   # 每步的時間

# 模擬股票價格的隨機過程
def simulate_stock_price(S0, r, sigma, T, N):
    dt = T / N
    prices = np.zeros(N)
    prices[0] = S0
    for i in range(1, N):
        dB = np.random.normal(0, np.sqrt(dt))  # 布朗運動增量
        prices[i] = prices[i-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * dB)
    return prices

# 生成股票價格路徑
stock_prices = simulate_stock_price(S0, r, sigma, T, N)

# 可視化股票價格
plt.figure(figsize=(10, 5))
plt.plot(np.linspace(0, T, N), stock_prices, label='Simulated Stock Price')
plt.title('Simulated Stock Price using Geometric Brownian Motion')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.grid()
plt.show()
