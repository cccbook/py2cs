from env import *
from inspect import signature

fstack = []
stack = []

class Function(object): # 函數定義
    def __init__(self, env):
        self.env = env
        self.narg = None
        self.entry = None

def push(o):
    global stack
    stack.append(o)

def pop():
    global stack
    return stack.pop()

def fpush(o):
    global fstack
    fstack.append(o)

def fpop():
    global fstack
    return fstack.pop()

def build():
    lines = code.split('\n')
    cmds = []
    for line in lines:
        if len(line)==0: continue
        cmd = line.split(' ')
        if len(cmd)<2:cmd.append('')
        print('cmd=', cmd)
        cmds.append(cmd)
    return cmds

def vm(code, env, mode="run"):
    cmds = build(code)
    pc = 0
    clen = len(cmds)
    print('pc=', pc, 'clen=', clen)
    while pc < clen:
        cmd = cmds[pc]
        print(f'{pc}:{cmd}')
        op, arg = cmd
        if op == "function":
            f = Function(env)
            f.narg = int(arg)
            f.entry = pc
            fpush(f)
            fstack.append(f)
            
        elif op == "fend":
            fstack.pop()
            pc = pop()

        # if mode == "define" and len(fstack) > 0: continue

        if op == "arg":
            env[arg] = stack.pop()
        elif op == "int":
            push(int(arg))
        elif op == "float":
            push(float(arg))
        elif op == "var":
            push(env.findVar(arg))
        elif op == "call":
            env = Env({}, env)
            f = pop()
            push(pc)
            pc = f.entry
            fpush(f)
        else:
            print('stack=', stack)
            fop = env.findVar(op)
            print('fop=', fop)
            sig = signature(fop)
            params = sig.parameters
            print('params=', params)
            args = []
            for param in params:
                arg = pop()
                print(f"{param}={arg}")
                args.append(arg)
            print('args=', args)
            fop(*args)

        pc += 1

code = """
int 3
int 4
function 2
arg x
arg y
var x
var y
+
fend
call
"""

vm(code, gEnv)