# 假設我們有一個投資組合的回報分佈
mean = 0.01  # 預期收益
std_dev = 0.02  # 標準差
simulations = 10000  # 模擬次數

# 模擬隨機回報
simulated_returns = np.random.normal(mean, std_dev, simulations)

# 計算95%信心水準下的VaR
VaR_mc = np.percentile(simulated_returns, 5)
print(f"Monte Carlo Simulation VaR (95%): {VaR_mc:.4f}")
