import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1)
x = np.linspace(-5, 5, 50)
ax.plot(x, norm.pdf(x))
plt.show()