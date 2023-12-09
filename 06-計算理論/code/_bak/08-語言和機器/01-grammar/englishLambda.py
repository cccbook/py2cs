import random

S = lambda: [NP(),VP()]
NP = lambda: [D(), N()]
VP = lambda: [V(), NP()]
D = lambda: choose(["a", "the"])
N = lambda: choose(["cat", "dog"])
V = lambda: choose(["chase", "eat"])
choose = lambda list:print(random.choice(list), end=" ")

S()
