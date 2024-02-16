g = globals()

def satisfy(exp, vars, values): # 測試 exp 在指令 vars[0..i]=values[0..i] 時，是否能被滿足。
    if len(values) == len(vars):
        for i in range(len(vars)):
            g[vars[i]] = values[i]
        result = eval(exp)
        print(values, '=>', result)
        if result: return values
        return None
    v0 = values.copy()
    v1 = values.copy()
    v0.append(0)
    v1.append(1)
    return satisfy(exp, vars, v0) or satisfy(exp, vars, v1)


def SAT(exp, vars) :
    print('exp=', exp)
    print(vars) 
    values = satisfy(exp, vars, [])
    return {'answer':values}


print(SAT('(x or y) and (not x or not z) and (x) and (y)', ['x', 'y', 'z']))
print(SAT('(x) and (not x) and (not y) and (not z)', ['x', 'y', 'z']))
