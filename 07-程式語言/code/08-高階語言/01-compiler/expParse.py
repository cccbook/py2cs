from random import *

"""
E = F ([+-] F)*
F = [0-9] | (E)
"""

def E():
    F()
    while isNext("+-"):
        next("+-")
        F()

def F():
    if isNext("("):
        next("(")
        E()
        next(")")
    else:
        next("0123456789")

def isNext(array):
    global ti, tokens
    if ti>=len(tokens): return False
    return tokens[ti] in array

def next(array):
    global ti
    if isNext(array):
        print(tokens[ti], end=" ")
        ti += 1
    else:
        print(f'錯誤: 詞彙 {tokens[ti]} 不在 {array} 中')

def parse(code):
    global ti, tokens
    ti=0
    tokens = list(code)
    E()

parse("3+(5-8)+7")
