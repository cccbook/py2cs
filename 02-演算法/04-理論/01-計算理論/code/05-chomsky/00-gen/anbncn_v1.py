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
        left = choice(lefts)
        i = rule.find(left)
        if i != -1:
            rights = G[left]
            right = choice(rights)
            rule = rule[0:i]+right+rule[i+len(left):]
            print(rule)
            if rule.islower(): break

gen(G)
