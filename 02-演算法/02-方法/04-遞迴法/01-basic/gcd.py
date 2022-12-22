def gcd(x, y):
	if x == 0: return y
	if y == 0: return x
	return gcd(y, x % y)

print('gcd(27, 45)=', gcd(27, 45))
