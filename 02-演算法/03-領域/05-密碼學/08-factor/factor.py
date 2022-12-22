import math

def factor(n) :
    for i in range(2, int(math.sqrt(n))):
        if n % i == 0: return i
    return -1

def isPrime(n):
    return factor(n) == -1

print('factor(15)=', factor(15))
print('factor(37)=', factor(37))
print('factor(9373467139)=', factor(9373467139))
# 以下大質數參考 《維基百科: 質數列表》 -- https://zh.wikipedia.org/wiki/%E8%B3%AA%E6%95%B8%E5%88%97%E8%A1%A8
print('factor(10000819)=', factor(10000819))
print('factor(3093215881333057)=', factor(3093215881333057))
print('factor(489133282872437289)=', factor(489133282872437289))
print('factor(489133282872437279)=', factor(489133282872437279))
print('factor(4776913109852041418248056622882488319)=', factor(4776913109852041418248056622882488319))
