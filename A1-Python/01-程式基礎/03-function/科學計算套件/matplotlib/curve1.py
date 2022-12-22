import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 1000)
y = np.sin(x)
z = np.cos(x)

plt.plot(x,y,label="$sin(x)$", color="red", linewidth=2)
plt.plot(x,z,label="$cos(x)$", color="blue", linewidth=1)

plt.show()