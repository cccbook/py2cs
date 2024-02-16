from parse import parse_lisp
import operator as op

Symbol = str
Number = (int, float)

class Env(dict):
    def __init__(self, vars, outer=None):
        self.outer = outer
        self.update(vars)

    def findEnv(self, var): # 找到最內層出現的環境變數
        if var in self: return self
        elif self.outer is None: raise Exception(var)
        else: return self.outer.findEnv(var)

    def findVar(self, var): # 找到最內層出現的環境變數
        env = self.findEnv(var)
        return env[var]


# 定義一些基本的運算符
ENV = {
    '+': op.add, '-': op.sub, '*': op.mul, '/': op.truediv,
    '>':op.gt, '<':op.lt, '>=':op.ge, '<=':op.le, '=':op.eq, 
    'abs': abs,
    'append':  op.add,  
    'apply':   lambda f, args: f(*args),
    'begin':   lambda *x: x[-1],
    'car':     lambda x: x[0],
    'cdr':     lambda x: x[1:], 
    'cons':    lambda x,y: [x] + y,
    'eq?':     op.is_, 
    'equal?':  op.eq, 
    'length':  len, 
    'list':    lambda *x: list(x), 
    'list?':   lambda x: isinstance(x,list), 
    'map':     map,
    'max':     max,
    'min':     min,
    'not':     op.not_,
    'null?':   lambda x: x == [], 
    'number?': lambda x: isinstance(x, Number),   
    'procedure?': callable,
    'round':   round,
    'symbol?': lambda x: isinstance(x, Symbol),
}


gEnv = Env(ENV)
