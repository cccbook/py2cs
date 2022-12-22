import numpy as np
import gd
import math

def sig(t):
    return 1.0/(1.0+math.exp(-t))

def relu(t):
    return t if t>0 else 0

def neuron(w, x, f=sig):
  return f(np.dot(w,x))
