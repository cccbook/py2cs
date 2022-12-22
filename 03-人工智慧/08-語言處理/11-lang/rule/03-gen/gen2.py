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

def choose(words, probs):
    i = r.randrange(len(words))
    return (words[i], probs[i])

def genWords(words, probs, p):
    word, prob = choose(words, probs)
    p['p'] = p['p'] * prob
    return [word]

def N(p):
    return genWords(['dog', 'cat'], [0.3, 0.7], p)

def V(p):
    return genWords(['chase', 'eat'], [0.6, 0.4], p)

def DET(p):
    return genWords(['a', 'the'], [0.5, 0.5], p)

p = {'p': 1.0}

for _ in range(10):
    print('%s %f'%(S(p), p['p']))
