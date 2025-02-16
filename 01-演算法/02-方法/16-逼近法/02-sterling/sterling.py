# https://en.wikipedia.org/wiki/Stirling%27s_approximation
import math

def factorial(n):
	r = 1
	for i in range(1, n+1):
		r *= i
	return r

def logFactorial(n):
	r = 0
	for i in range(1,n+1):
		r += math.log(i)
	return r

def stirling(n):
    return n*math.log(n)-n+1


a = [3, 10, 100, 1000]

for n in a:
    print(f'{n}!={factorial(n)}')

for n in a:
    print(f'{n}:{logFactorial(n)} {stirling(n)} ratio={logFactorial(n)/stirling(n)}')
