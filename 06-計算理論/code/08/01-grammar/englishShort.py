import random

def S(): NP(),VP()
def NP(): D(), N()
def VP(): V(), NP()
def D(): choose(["a", "the"])
def N(): choose(["cat", "dog"])
def V(): choose(["chase", "eat"])
def choose(list):print(random.choice(list), end=" ")

S()