from functools import reduce

a = range(1,5)
print('a=', a)
print('list(a)=', list(a))
print('map(a,x^2)=', list(map(lambda x:x*x, a)))
print('filter(a, 奇數)=', list(filter(lambda x:x%2==1, a)))
print('reduce(x+y, a)=', reduce(lambda x,y:x+y, a))
