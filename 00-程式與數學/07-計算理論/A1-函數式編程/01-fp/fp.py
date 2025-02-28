from functools import reduce

a = range(1,5)
print(list(map(lambda x:x*x, a)))
print(list(filter(lambda x:x%2==1, a)))
print(reduce(lambda x,y:x+y, a))
