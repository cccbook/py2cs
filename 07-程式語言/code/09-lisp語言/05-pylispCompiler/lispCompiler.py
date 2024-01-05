from parse import parse_lisp

def emit(code):
    print(code)

def gen(o):
    if isinstance(o, list):
        op = o[0]
        # print('op=', op, 'type(o)=', type(o))
        if op == 'lambda':
            args = o[1]
            body = o[2]
            # print('args=', args)
            # print('body=', body)
            emit("lambda")
            for arg in args:
                emit(f"arg:{arg}")
            gen(body)
            emit("-lambda")
        else:
            for arg in o[1:]:
                gen(arg)
            emit(f"{op}")
    else:
        emit(f"push {o}")

def compile(code):
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
