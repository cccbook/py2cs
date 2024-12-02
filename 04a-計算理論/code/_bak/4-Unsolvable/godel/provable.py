# 哥德爾不完備定理：不存在一個程式，可以正確判斷一個「包含算術的一階邏輯字串」是否為定理。
def Proveable(s):
    if isTheorem(s): 
        return 1
    else:
        return 0

isTheorem('∃s -Provable(s) & -Provable(-s)') # = ?
