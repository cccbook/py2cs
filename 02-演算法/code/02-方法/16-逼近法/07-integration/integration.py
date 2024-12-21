# import sys

def integrate(f, a, b, h=0.001):
	area = 0
	x = a
	while x<b:
		area += f(x)*h
		x+=h
	return area

print(f'integrate(x**2, 0, 1)=', integrate(lambda x:x**2, 0, 1))
