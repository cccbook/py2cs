import random as r

'''
S = NP VP
NP = DET N
VP = V NP
N = 狗 | 貓
V = 追 | 吃
DET = 一隻 | 這隻
'''

def S():
    return NP() + ' ' + VP()

def NP():
    return DET() + ' ' + N()

def VP():
    return V() + ' ' + NP()

def N():
    return r.choice(['狗', '貓'])

def V():
    return r.choice(['追', '吃'])

def DET():
    return r.choice(['一隻', '這隻'])

print(S())
