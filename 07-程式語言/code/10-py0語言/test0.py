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

e2c = {'a':'一隻', 'dog':'狗', 'chase':'追', 'cat':'貓'}
print('e2c=', e2c)

def mt(s):
	r = []
	words = s.split(' ')
	for e in words:
		c = e2c[e]
		r.append(c)
	return ' '.join(r)

print("fib(5)=", fib(5))
print('sum(10)=', sum(10))
print('mt(a dog chase a cat)=', mt('a dog chase a cat'))

a = [1,2,3]
print('a=', a)

'''