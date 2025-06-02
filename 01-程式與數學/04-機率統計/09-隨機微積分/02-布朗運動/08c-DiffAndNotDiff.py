import numpy as np
import matplotlib.pyplot as plt

# 時間參數
T = 1.0
N = 100000
dt = T / N
t = np.linspace(0, T, N)

# --- 模擬布朗運動 ---
B = np.cumsum(np.sqrt(dt) * np.random.randn(N))

# --- 可微函數（sin）的定義 ---
f = np.sin(2 * np.pi * t)

# 選定觀察點 t0 = 0.5
t0 = 0.5
i0 = int(t0 / dt)

# 計算不同 h 下的變化率
hs = np.logspace(-5, -1, 30)
brownian_rates = []
smooth_rates = []

for h in hs:
    di = int(h / dt)
    if i0 + di < len(t):
        # 布朗運動變化率
        dB = (B[i0 + di] - B[i0]) / h
        brownian_rates.append(np.abs(dB))

        # 可微函數變化率
        df = (f[i0 + di] - f[i0]) / h
        smooth_rates.append(np.abs(df))
    else:
        brownian_rates.append(np.nan)
        smooth_rates.append(np.nan)

# --- 繪圖比較 ---
plt.figure(figsize=(9, 5))
plt.plot(hs, brownian_rates, label='Brown Movement', marker='o')
plt.plot(hs, smooth_rates, label='Differentable sin(2πt)', marker='x')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('h（Time Span）')
plt.ylabel(r'|Change Rate| = $|\frac{f(t+h) - f(t)}{h}|$')
plt.title('Differentable vs Brown Movement Compare')
plt.grid(True, which='both')
plt.legend()
plt.tight_layout()
plt.show()
