# 參考 -- https://realpython.com/python-histograms/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Generate data on commute times.
size, scale = 1000, 10
commutes = pd.Series(np.random.gamma(scale, size=size) ** 1.5)

commutes.plot.hist(grid=True, bins=20, rwidth=0.9, color='#607c8e')

plt.title('Commute Times for 1,000 Commuters')
plt.xlabel('Counts')
plt.ylabel('Commute Time')
plt.grid(axis='y', alpha=0.75)

plt.show()

