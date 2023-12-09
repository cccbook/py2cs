from random import *

"""
E = F ([+-] F)*
F = [0-9] | (E)
"""

opAsm = {"+":"add", "-":"sub"}

def E():
    F()
    while isNext("+-"):
        op = next("+-")
        F()
        print(f"{opAsm[op]}")

def F():
    if isNext("("):
        next("(")
        E()
        next(")")
    else:
        d = next("0123456789")
        print(f"push {d}")

def isNext(array):
    global ti, tokens
    if ti>=len(tokens): return False
    return tokens[ti] in array

def next(array):
    global ti
    if isNext(array):
        token = tokens[ti]
        ti += 1
        return token
    else:
        print('======== error ===========')
        print(''.join(tokens))
        print(f"{ti*' '}^")
        print(f'錯誤: 位置 {ti} 應該是 {array} 的元素，但卻發現 {tokens[ti]}')
        raise
        

def parse(code):
    global ti, tokens, tempTop
    ti, tokens, tempTop = 0, list(code), 0
    print(code)
    E()

parse("3+(5-8)+7")
parse("3+(5-8+)+7")
