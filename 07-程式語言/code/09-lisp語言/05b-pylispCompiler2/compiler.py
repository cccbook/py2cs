from parse import parse_lisp
from cenv import Env, gEnv

fnTop = 0
envStack = []

def emit(code):
    print(code)

def gen(o):
    global env
    if isinstance(o, list):
        op = o[0]
        if op == 'fn':
            name = o[1]
            args = o[2]
            body = o[3]
            env = Env(zip(args, [0]*len(args)), env)
            envStack.append(env)
            emit(f"fn {name}")
            for arg in args:
                emit(f"arg {arg}")
            gen(body)
            env = envStack.pop()
            emit("fend")
        elif op == 'var':
            name = o[1]
            value = o[2]
            env[name] = 0
            gen(value)
            emit(f"var {name} {value}")
        else:
            for arg in o[1:]:
                gen(arg)
            if isinstance(op, list):
                gen(op)
                if op[0] == 'fn':
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
                emit(f"get {o} # env.id={e.id}")
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
    compile("((fn add (x y) (+ x y)) 3 4)")
    compile("(if (> 6 5) (+ 1 1) (+ 2 2))")
    # compile("(begin (define x 1) (set! x (+ x 1)) (+ x 1))")
    compile("(fn twice (x) (* 2 x))")
    compile("(fn sub3 (x) (var t 3) (- x t) (return))")
