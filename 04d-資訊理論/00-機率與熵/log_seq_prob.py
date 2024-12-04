import math

def log2(x):
    return math.log(x, 2)

p = 0.25

print('p=', p)
print('10次 p 事件的機率=p**10=', p**10)
print('1000次 p 事件的機率=p**1000=', p**1000)

print()
print('log2(1/p)=', log2(1/p))
print('log(10次 p 事件的機率)=10*log2(1/p)=', 10*log2(1/p))
print('log(100次 p 事件的機率)=100*log2(1/p)=', 1000*log2(1/p))
