import numpy as np
import matplotlib.pyplot as plt

dt = 0.001
T = 1.0
N = int(T/dt)
t = np.linspace(0, T, N)
B = np.cumsum(np.sqrt(dt) * np.random.randn(N))

plt.plot(t, B, label='Brown Movement B(t)')
plt.xlabel('Time t')
plt.ylabel('B(t)')
plt.title('Brown Movement')
plt.grid(True)
plt.legend()
plt.show()
