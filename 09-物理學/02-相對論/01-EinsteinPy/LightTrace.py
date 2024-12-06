from einsteinpy.coordinates import SphericalDifferential
from einsteinpy.symbolic import Schwarzschild
from einsteinpy import static
from sympy import symbols
import numpy as np
import matplotlib.pyplot as plt

# 設置事件座標
r, t = symbols('r t')

# Schwarzschild 度量
schwarzschild_metric = Schwarzschild()

# 計算光線的軌跡
light_ray = schwarzschild_metric.null_geodesic(t, r)

# 可視化光線的路徑
plt.plot(np.linspace(0, 10, 100), light_ray)
plt.title('Light Ray Trajectory around a Schwarzschild Black Hole')
plt.xlabel('Time (s)')
plt.ylabel('Radius (m)')
plt.show()
