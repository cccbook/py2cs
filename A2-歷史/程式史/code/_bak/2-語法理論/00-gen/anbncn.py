# https://en.wikipedia.org/wiki/Context-sensitive_language

from random import *

G = {
    "S":["aBC","aSBC","aSBC","aSBC"], # 產生 anBnCn
    "CB":["CZ"], # CB=>CZ=>WZ=>WC=>BC 導致 C 和 B 交換了?
    "CZ":["WZ"],
    "WZ":["WC"],
    "WC":["BC"],
    "aB":["ab"], # 換成小寫
    "bB":["bb"],
    "bC":["bc"],
    "cC":["cc"]
}

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

gen(G)
