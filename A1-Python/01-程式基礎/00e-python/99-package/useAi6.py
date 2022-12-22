import ai6.nn as nn

def f(p):
	[x,y] = p
	return x*x + y*y

p = [1.0, 3.0]

nn.gradientDescendent(f, p)
