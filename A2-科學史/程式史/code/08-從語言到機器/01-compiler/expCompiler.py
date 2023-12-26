from random import *

"""
E = F ([+-] F)*
F = [0-9] | (E)
"""

def newTemp():
    global tempTop
    tempTop += 1
    return f"T{tempTop}"

def E():
    f = F()
    while isNext("+-"):
        op = next("+-")
        f2 = F()
        t = newTemp()
        print(f"{t} = {f} {op} {f2}")
        f = t
    return f

def F():
    if isNext("("):
        next("(")
        e = E()
        next(")")
        return e
    else:
        return next("0123456789")

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

def compile(code):
    global ti, tokens, tempTop
    ti, tokens, tempTop = 0, list(code), 0
    print(code)
    E()

compile("3+(5-8)+7")
# compile("3+(5-8+)+7")