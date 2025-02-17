import autograd.numpy as np  # Thinly-wrapped numpy
from autograd import grad    # The only autograd function you may ever need

def tanh(x):                 # Define a function
    y = np.exp(-2.0 * x)
    return (1.0 - y) / (1.0 + y)

grad_tanh = grad(tanh)       # Obtain its gradient function
g1 = grad_tanh(1.0)
print('g1=', g1)               # Evaluate the gradient at x = 1.0

g2 = (tanh(1.0001) - tanh(0.9999)) / 0.0002 
print('g2=', g2)
