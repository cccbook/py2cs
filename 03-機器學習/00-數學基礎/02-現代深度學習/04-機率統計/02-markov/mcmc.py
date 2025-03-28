from prob import P
from random import random
def mcmc(s): # Monte Carlo Markov Chain
    if random() > P[s[0]]: return 0
    for i in range(len(s)-1):
        key = f'{s[i]}=>{s[i+1]}'
        if random() > P[key]: return 0
    return 1

def markov(s, n):
    passCount = 0
    for _ in range(n):
        passCount += mcmc(s) 
    return passCount/n

seq = ['b', 'a', 'b', 'b']

print(f'P({seq})={markov(seq, 100000)}')
