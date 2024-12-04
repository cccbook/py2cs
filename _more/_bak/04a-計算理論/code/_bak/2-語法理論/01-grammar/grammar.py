from random import *

def gen(G):
    lefts = list(G.keys())
    rule = "S"
    while True:
        for left in lefts:
            i = rule.find(left)
            if i != -1:
                rights = G[left]
                # right = choice(rights)
                ri = randrange(0, len(rights)); right = rights[ri]
                rule = rule[0:i]+right+rule[i+len(left):]
                print(rule)
                if rule.islower(): return rule
                break
