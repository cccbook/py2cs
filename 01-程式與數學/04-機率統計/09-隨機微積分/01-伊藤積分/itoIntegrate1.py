import numpy as np
import matplotlib.pyplot as plt

# 參數
T = 1.0       # 時間總長
N = 1000      # 分割數
dt = T / N    # 每步時間
t = np.linspace(0, T, N+1)

# 模擬布朗運動 B_t
np.random.seed(0)
dB = np.sqrt(dt) * np.random.randn(N)
B = np.concatenate(([0], np.cumsum(dB)))

# 數值估算伊藤積分 I_t = ∫ B_s dB_s
# 使用左端點法則: sum(B_i * dB_i)
ito_integral = np.sum(B[:-1] * dB)

# 理論值: 0.5 * B_T^2 - 0.5 * T
ito_exact = 0.5 * B[-1]**2 - 0.5 * T

# 輸出結果
print(f"模擬伊藤積分值： {ito_integral:.5f}")
print(f"理論值：        {ito_exact:.5f}")
print(f"誤差：          {abs(ito_integral - ito_exact):.5f}")

# 可視化布朗運動
plt.plot(t, B)
plt.title("Simulate Ito Integrateß $B_t$")
plt.xlabel("Time t")
plt.ylabel("B(t)")
plt.grid(True)
plt.show()
