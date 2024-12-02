from parse import parse_lisp
from env import Env, gEnv

lambdaTop = 0
fstack = []

def emit(code):
    print(code)

def gen(o):
    global env
    if isinstance(o, list):
        op = o[0]
        if op == 'lambda':
            args = o[1]
            body = o[2]
            fenv = Env(zip(args, [None]*len(args)), env)
            env = fenv
            f = {"fid":lambdaTop, "env":fenv}
            fstack.append(f)
            emit(f"function")
            for arg in args:
                emit(f"arg {arg}")
            gen(body)
            f = fstack.pop()
            env = f['env']
            emit("fend")
        elif op == 'define':
            name = o[1]
            body = o[2]
            env[name] = None
            gen(body)
            emit(f"var {name}")
            emit(f"define")
        else:
            for arg in o[1:]:
                gen(arg)
            if isinstance(op, list):
                gen(op)
                if op[0] == 'lambda':
                    emit("call")
            else:
                emit(f"{op}")
    else:
        if isinstance(o, int):
            emit(f"int {o}")
        elif isinstance(o, float):
            emit(f"float {o}")
        elif isinstance(o, str):
            e = env.findEnv(o)
            if e:
                emit(f"var {o} # env.id={e.id}")
            else:
                raise Exception(f"Error: variable {o} not found!")
        else:
            emit(f"push {o}")

def compile(code):
    global env
    env = gEnv
    emit(f"compile:{code}")
    o = parse_lisp(code)
    print('o=', o)
    return gen(o)

# 測試程式碼轉換
if __name__ == "__main__":
    compile("(+ 2 (* 3 4))")
    compile("((lambda (x y) (+ x y)) 3 4)")
    compile("(if (> 6 5) (+ 1 1) (+ 2 2))")
    compile("(begin (define x 1) (set! x (+ x 1)) (+ x 1))")
    compile("(define twice (lambda (x) (* 2 x)))")
