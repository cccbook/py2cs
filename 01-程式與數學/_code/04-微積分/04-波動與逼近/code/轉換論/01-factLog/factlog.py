import math

def logFactorial(n):
	r = 0
	for i in range(1,n+1):
		r += math.log(i)
	return r

def factorial(n):
	logf = logFactorial(n)
	return int(math.exp(logf))

print(factorial(10))
