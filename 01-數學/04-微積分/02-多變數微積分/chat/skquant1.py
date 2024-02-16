import numpy as np
from skquant.opt import minimize

# some interesting objective function to minimize
def objective_function(x):
     fv = np.inner(x, x)
     fv *= 1 + 0.1*np.sin(10*(x[0]+x[1]))
     return np.random.normal(fv, 0.01)

# create a numpy array of bounds, one (low, high) for each parameter
bounds = np.array([[-1, 1], [-1, 1]], dtype=float)

# budget (number of calls, assuming 1 count per call)
budget = 40

# initial values for all parameters
x0 = np.array([0.5, 0.5])

# method can be ImFil, SnobFit, Orbit, NOMAD, or Bobyqa
result, history = minimize(objective_function, x0, bounds, budget, method='imfil')
print('result=', result)
print('history=', history)
