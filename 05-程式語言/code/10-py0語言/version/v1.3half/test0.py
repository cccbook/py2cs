code = '''
def fib(n):
	if n == 0 or n == 1:
		a = 3
		return 1
	return fib(n-1)+fib(n-2)

def sum(n):
	s = 0
	i = 1
	while i<n:
		s = s+i
		i = i+1
	return s

def sum2(n):
	s = 0
	for i in range(1,n):
		s = s + i
	return s

print("fib(5)=", fib(5))
print('sum(10)=', sum(10))
print('sum2(10)=', sum2(10))
'''