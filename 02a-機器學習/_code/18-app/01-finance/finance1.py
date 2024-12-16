import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 模擬股票收益
np.random.seed(42)
returns = np.random.normal(0.01, 0.02, 1000)  # 假設收益率為正態分佈

# 計算95%信心水準下的VaR
VaR_95 = np.percentile(returns, 5)

# 顯示結果
print(f"95% Value at Risk (VaR): {VaR_95:.4f}")

# 繪製收益分佈
plt.hist(returns, bins=50, edgecolor='black')
plt.axvline(VaR_95, color='r', linestyle='dashed', linewidth=2)
plt.title("Simulated Stock Returns and VaR (95%)")
plt.show()


# 計算夏普比率
Rf = 0.002  # 假設無風險回報率為 0.2%
Rp = np.mean(returns)  # 假設投資回報為平均收益
sigma_p = np.std(returns)  # 計算回報的標準差

sharpe_ratio = (Rp - Rf) / sigma_p
print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
