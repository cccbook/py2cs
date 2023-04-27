import random as r

'''
S = NP VP
NP = DET N
VP = V NP
N = dog | cat
V = chase | eat
DET = a | the
'''

def S():
    return NP() + ' ' + VP()

def NP():
    return DET() + ' ' + N()

def VP():
    return V() + ' ' + NP()

def N():
    return r.choice(['dog', 'cat'])

def V():
    return r.choice(['chase', 'eat'])

def DET():
    return r.choice(['a', 'the'])

print(S())
