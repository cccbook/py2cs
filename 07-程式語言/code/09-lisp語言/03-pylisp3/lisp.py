from parse import parse_lisp
import operator as op

isa = isinstance
Symbol = str

class Env(dict):
    def __init__(self, vars, outer=None):
        self.outer = outer
        self.update(vars)

    def find(self, var): # 找到最內層出現的環境變數
        if var in self: return self.get(var)
        elif self.outer is None: raise Exception(var)
        else: return self.outer.find(var)

class Function(object): # 函數定義
    def __init__(self, params, body, env):
        self.params, self.body, self.env = params, body, env
    def __call__(self, *args): 
        if len(args) != len(self.params):
            raise Exception(f'({self.params}) 和 {args} 參數數量不符!')
        fenv = Env(zip(self.params, args), self.env)
        return evaluate(self.body, fenv)

# 定義一些基本的運算符
ENV = Env({ '+': op.add, '-': op.sub, '*': op.mul, '/': op.truediv })

# LISP (Scheme) 解釋器
def evaluate(exp, env={}):
    if isa(exp, list):
        if exp[0] == 'define':
            _, var, value = exp
            env[var] = evaluate(value, env)
        elif exp[0] == 'lambda':
            _, params, body = exp
            return Function(params, body, env)
        else: 
            # 函數調用
            op = evaluate(exp[0], env)
            args = [evaluate(arg, env) for arg in exp[1:]]
            return op(*args)
    elif isa(exp, str) and (env.find(exp)):
        # 變數引用
        return env.find(exp)
    else:
        # 常數
        return exp

def run(blocks):
    env = Env({ '+': op.add, '-': op.sub, '*': op.mul, '/': op.truediv })
    for block in blocks:
        code = parse_lisp(block)
        print(code)
        result = evaluate(code, env)
        print(result)

# 測試 Scheme 解釋器
if __name__ == "__main__":
    # 定義一個 lambda 函數並調用
    run(['((lambda (x y) (+ x y)) 3 4)'])
    
    # 定義一個有名稱的函數並調用
    run([
    "(define add (lambda (x y) (+ x y)))",
    "(add 3 4)"
    ])
