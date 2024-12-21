from random import *

def gen(G):
    lefts = list(G.keys())
    rule = " S "
    while True:
        for left0 in lefts:
            left = f' {left0} '
            i = rule.find(left)
            if i != -1:
                rights = G[left0]
                # right = choice(rights)
                ri = randrange(0, len(rights));
                right = f' {rights[ri]} '
                rule = rule[0:i]+right+rule[i+len(left):]
                print(rule)
                if rule == rule.lower(): return rule
                break


