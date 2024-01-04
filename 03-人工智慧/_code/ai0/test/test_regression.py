import matplotlib.pyplot as plt
import numpy as np
from optimize import gd
from loss import mse_loss

x = np.array([0, 1, 2, 3, 4], dtype=np.float32)
y = np.array([1.9, 3.1, 3.9, 5.0, 6.2], dtype=np.float32)
# y = np.array([2, 3, 4, 5, 6], dtype=np.float32)

import sys

def predict(p, xt):
    return p[0]+p[1]*xt

def loss(p):
    return mse_loss(p, predict, x, y)

if len(sys.argv) > 1 and sys.argv[1] == "micrograd":
    import nn
    p = [nn.Value(0.0), nn.Value(0.0)]
    model = nn.Vars(p)
    plearn = nn.gd(p, loss, model, max_loops=3000, dump_period=1)
else:        
    p = [0.0, 0.0]
    plearn = gd(p, loss, max_loops=3000, dump_period=1)

# Plot the graph
y_predicted = list(map(lambda t: plearn[0]+plearn[1]*t, x))
print('y_predicted=', y_predicted)
plt.plot(x, y, 'ro', label='Original data')
plt.plot(x, y_predicted, label='Fitted line')
plt.legend()
plt.show()
