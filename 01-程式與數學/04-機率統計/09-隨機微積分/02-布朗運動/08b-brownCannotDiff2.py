import numpy as np
import matplotlib.pyplot as plt

# 模擬布朗運動
T = 1.0
N = 100000  # 數值解析度
dt = T / N
t = np.linspace(0, T, N)
B = np.cumsum(np.sqrt(dt) * np.random.randn(N))

# 選定 t0 = 0.5，找出對應索引
t0 = 0.5
i0 = int(t0 / dt)

# 測試不同 h 對應的變化率
hs = np.logspace(-5, -1, 30)  # h 從 10^-5 到 10^-1
rates = []

for h in hs:
    di = int(h / dt)
    if i0 + di < len(B):
        rate = (B[i0 + di] - B[i0]) / h
        rates.append(np.abs(rate))  # 取絕對值便於觀察大小
    else:
        rates.append(np.nan)

# 繪圖
plt.figure(figsize=(8, 5))
plt.plot(hs, rates, marker='o')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('h（time span）')
plt.ylabel(r'|Change Rate| = $| \frac{B(t+h) - B(t)}{h} |$')
plt.title('Brown Movement when h => 0 will not converge to a limit')
plt.grid(True, which='both')
plt.show()
