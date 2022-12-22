def e(n):
	return (1+1.0/n)**n

for i in range(100):
	n = (i+1)*100.0
	print('n=', n, 'e(n)=', e(n))
