import sys
import operator as op
from parse import parse_lisp

class Symbol(str): pass
def is_pair(x): return x != [] and isa(x, list)
def cons(x, y): return [x]+y
isa = isinstance

# 定義一些基本的運算符
ENV = {
    '+': op.add, '-': op.sub, '*': op.mul, '/': op.truediv,
    'not':op.not_, '>':op.gt, '<':op.lt, '>=':op.ge, '<=':op.le, '=':op.eq, 
    'equal?':op.eq, 'eq?':op.is_, 'length':len, 'cons':cons,
    'car':lambda x:x[0], 'cdr':lambda x:x[1:], 'append':op.add,
    'list':lambda *x:list(x), 'list?': lambda x:isa(x,list),
    'null?':lambda x:x==[], # 'symbol?':lambda x: isa(x, Symbol),
    'boolean?':lambda x: isa(x, bool), 'pair?':is_pair, 
    # 'port?': lambda x:isa(x,file), 
    'apply':lambda proc,l: proc(*l), 
    # 'eval':lambda x: eval(expand(x)), 'load':lambda fn: load(fn), 'call/cc':callcc,
    'open-input-file':open,'close-input-port':lambda p: p.file.close(), 
    'open-output-file':lambda f:open(f,'w'), 'close-output-port':lambda p: p.close(),
    # 'eof-object?':lambda x:x is eof_object, 'read-char':readchar,
    # 'read':read, 'write':lambda x,port=sys.stdout:port.write(to_string(x)),
    'display':lambda x,port=sys.stdout:port.write(x if isa(x,str) else to_string(x))
}

def list2str(x):
    if x is True: return "#t"
    elif x is False: return "#f"
    elif isa(x, Symbol): return x
    elif isa(x, str): return '"%s"' % x.encode('string_escape').replace('"',r'\"')
    elif isa(x, list): return '('+' '.join(map(to_string, x))+')'
    elif isa(x, complex): return str(x).replace('j', 'i')
    else: return str(x)

# Scheme 解釋器
def evaluate(exp, env={}):
    if isinstance(exp, list):
        # 特殊形式
        if exp[0] == 'define':
            _, var, value = exp
            env[var] = evaluate(value, env)
        elif exp[0] == 'lambda':
            _, parameters, body = exp
            return lambda *args: evaluate(body, dict(zip(parameters, args)), env)
        else:
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
    text = "(+ 2 (* 3 4))"
    code = parse_lisp(text)
    print('code=', code)
    # code = ['+', 2, ['*', 3, 4]]
    # 解釋和計算結果
    result = evaluate(code, ENV)
    print(result)
"""
    # 定義一個 lambda 函數並調用
    scheme_lambda = ['(lambda (x y) (+ x y))', 3, 4]
    lambda_result = evaluate(scheme_lambda, ENV)
    print(lambda_result)
"""