import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# 參數
L = 10  # 空間長度
T = 2   # 時間
dx = 0.1
dt = 0.01
nx = int(L / dx) + 1
nt = int(T / dt) + 1
# nt = 5*int(T / dt) + 1

x = np.linspace(-L / 2, L / 2, nx)
psi = np.exp(-x**2) + 1j * np.zeros(nx)  # 初始波函數
V = 0.5 * x**2  # 簡單的勢場（諧振子）
hbar = 1.0
m = 1.0

# 數值解
for n in range(nt):
    # 演化：Trotter 分裂步法
    psi = np.exp(-1j * V * dt / hbar / 2) * psi
    psi_k = np.fft.fft(psi)  # 轉到動量空間
    k = 2 * np.pi * np.fft.fftfreq(nx, dx)
    psi_k = np.exp(-1j * hbar * k**2 / (2 * m) * dt) * psi_k
    psi = np.fft.ifft(psi_k)  # 回到實空間
    psi = np.exp(-1j * V * dt / hbar / 2) * psi

# 繪圖
plt.plot(x, np.abs(psi)**2, label="|ψ(x)|²")
plt.xlabel("x")
plt.ylabel("|ψ(x)|²")
plt.title("薛定諤方程數值解")
plt.legend()
plt.show()
