import operator as op

# 定義一些基本的運算符
ENV = { '+': op.add, '-': op.sub, '*': op.mul, '/': op.truediv }

# Scheme 解釋器
def evaluate(exp, env={}):
    if isinstance(exp, list):
        # 函數調用
        op = evaluate(exp[0], env)
        args = [evaluate(arg, env) for arg in exp[1:]]
        return op(*args)
    elif isinstance(exp, str) and (exp in env):
        # 變數引用
        return env[exp]
    else:
        # 常數
        return exp

# 測試 Scheme 解釋器
if __name__ == "__main__":
    # 定義一個簡單的 Scheme 表達式
    code = ['+', 2, ['*', 3, 4]]
    # 解釋和計算結果
    result = evaluate(code, ENV)
    print(code, '=', result)
