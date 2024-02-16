import random

LEVEL = 0

def BNF(func):
    def warp():
        global LEVEL
        print(f"{LEVEL*' '}+{func.__name__}")
        LEVEL += 1
        func()
        LEVEL -= 1
        print(f"{LEVEL*' '}-{func.__name__}")
    return warp

@BNF
def S(): NP(),VP()

@BNF
def NP(): D(), N()

@BNF
def VP(): V(), NP()

@BNF
def D(): choose(["a", "the"])

@BNF
def N(): choose(["cat", "dog"])

@BNF
def V(): choose(["chase", "eat"])

def choose(list):print(f"{LEVEL*' '}{random.choice(list)}")

S()
