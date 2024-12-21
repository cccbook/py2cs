from pair import *

def make_rat(n, d): return pair(n, d)
def numer(x): return head(x)
def denom(x): return tail(x)

def print_rat(x):
    return print(f"{numer(x)}/{denom(x)}")

def add_rat(x, y):
    return make_rat(
        numer(x) * denom(y) + numer(y) * denom(x),
        denom(x) * denom(y)
    )

def sub_rat(x, y):
    return make_rat(
        numer(x) * denom(y) - numer(y) * denom(x),
        denom(x) * denom(y)
    )

def mul_rat(x, y):
    return make_rat(
        numer(x) * numer(y),
        denom(x) * denom(y)
    )

def div_rat(x, y):
    return make_rat(
        numer(x) * denom(y),
        denom(x) * numer(y)
    )

def equal_rat(x, y):
    return numer(x) * denom(y) == numer(y) * denom(x)
