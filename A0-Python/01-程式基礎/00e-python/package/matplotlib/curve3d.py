import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

x,y = np.mgrid[-2:2:20j, -2:2:20j]
z = x*np.exp(-x**2-y**2)

ax = plt.subplot(111, projection='3d')
ax.plot_surface(x,y,z, rstride=2, cstride=1) # , cmap=plt.cm.Blues

plt.show()
