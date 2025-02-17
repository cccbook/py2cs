from prob import P
from random import random

def mcmc(): # Monte Carlo Markov Chain
    s1 = 'a' if (random() < P['a']) else 'b' #  隨機選定一個初始狀態 (s1 初始狀態是 a 還是 b)
    s2 = 'a' if (random() < P[s1+'=>'+'a']) else 'b' #  從初始狀態會轉到的下一個狀態
    if s1 == s2: return #  如果沒有轉移，那麼不影響機率
    P[s1] -= 0.0001 #  否則、初始狀態的機率減小一點點
    P[s2] += 0.0001 #  然後、下一個狀態的機率增加一點點。

def gibbs(n):
    for _ in range(n):
        mcmc()

print("P=", P)
gibbs(1000000)
print('P=', P)

