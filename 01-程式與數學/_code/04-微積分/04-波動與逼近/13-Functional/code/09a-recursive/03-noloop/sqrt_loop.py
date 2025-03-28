from math import *

def sqrt(x):
	low = 1.0
	high = x
	while True:
		mid = (low+high)/2.0
		if abs(mid*mid-x)<0.0001:
			return mid
		elif mid*mid>x:
			high = mid
		elif mid*mid<x:
			low = mid

print(sqrt(3.0))
