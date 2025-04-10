import math

def logFactorial(n):
	r = 0
	for i in range(1,n+1):
		r += math.log(i)
	return r

def factorial(n):
	logf = logFactorial(n)
	print('logf=', logf)
	r = math.exp(logf)
	print('r=', r)
	return int(r)

print(factorial(10))
