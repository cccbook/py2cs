def df(f, x, h=0.001):
	return (f(x+h)-f(x))/h

def power3(x):
	return x**3

print('df(power3, 2)=', df(power3, 2))
