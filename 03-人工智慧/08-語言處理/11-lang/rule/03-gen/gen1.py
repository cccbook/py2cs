import random as r

'''
S = NP VP
NP = DET N
VP = V NP
N = dog | cat
V = chase | eat
DET = a | the
'''

def S(p):
    p['p'] = 1.0
    return NP(p) + VP(p)

def NP(p):
    return DET(p) + N(p)

def VP(p):
    return V(p) + NP(p)

def N(p):
    p['p'] = p['p'] * 0.5
    return r.choices(['dog', 'cat'], [0.3, 0.7])

def V(p):
    p['p'] = p['p'] * 0.5
    return r.choices(['chase', 'eat'], [0.6, 0.4])

def DET(p):
    p['p'] = p['p'] * 0.5
    return r.choices(['a', 'the'], [0.5, 0.5])

p = {'p': 1.0}

for _ in range(10):
    print('%s %f'%(S(p), p['p']))
