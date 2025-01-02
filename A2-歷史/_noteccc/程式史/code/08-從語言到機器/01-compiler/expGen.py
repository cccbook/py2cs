from random import *

def E():
    if random() < 0.5:
        T()  
    else:
        print("(", end="")
        E()
        OP()
        E()
        print(")", end="")

def OP():
    choose(["+", "-"])

def T():
    choose("0123456789")

def choose(list):
    print(f"{choice(list)}", end="")

E()
