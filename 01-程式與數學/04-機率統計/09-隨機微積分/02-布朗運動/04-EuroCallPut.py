import numpy as np

# 參數設定
T = 1.0            # 到期時間
N = 1000           # 時間步數
M = 100000         # 模擬路徑數
dt = T / N

mu = 0.1           # 漂移率（不用於選擇權定價，可忽略）
sigma = 0.2        # 波動率
r = 0.05           # 無風險利率
S0 = 100           # 初始價格
K = 110            # 履約價格

# 模擬 S_T（使用 GBM 解析解）
np.random.seed(42)
Z = np.random.randn(M)
S_T = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)

# 計算歐式選擇權價格（折現期望值）
call_payoff = np.maximum(S_T - K, 0)
put_payoff = np.maximum(K - S_T, 0)

call_price = np.exp(-r * T) * np.mean(call_payoff)
put_price = np.exp(-r * T) * np.mean(put_payoff)

# 輸出結果
print(f"歐式 Call 價格：{call_price:.4f}")
print(f"歐式 Put  價格：{put_price:.4f}")
