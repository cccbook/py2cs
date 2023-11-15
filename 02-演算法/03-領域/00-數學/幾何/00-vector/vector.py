import math

def cmul(c, a):
	r = a.copy()
	for i in range(len(a)):
		r[i] = c*r[i]
	return r

def neg(a):
	return cmul(-1.0, a)

def add(a,b):
	r = [0]*len(a)
	for i in range(len(a)):
		r[i] = a[i]+b[i]
	return r

def sub(a,b):
	return add(a, neg(b))

def dot(a,b):
	r = 0.0
	for i in range(len(a)):
		r += a[i]*b[i]
	return r

def norm(a):
	r = 0.0
	for x in a:
		r += x**2
	return math.sqrt(r)

length = norm

def distance(a,b):
	return length(sub(a,b))

def cross(a,b):
	

if __name__ == '__main__':
	v0 = [0.0,0.0]; v1 = [1.0,1.0]; v2 = [2.0, 1.0]; v3 = [-1.0, 0.0]
	print('v0=', v0, 'v1=', v1, 'v2=', v2, 'v3=', v3)
	print('neg(v1)=', neg(v1))
	print('add(v1,v2)=', add(v1,v2))
	print('sub(v1,v2)=', sub(v1,v2))
	print('dot(v1,v2)=', dot(v1,v2))

	# print('cross(v1, v2)=', cross(v1, v2))
