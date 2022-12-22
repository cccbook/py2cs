# If  = lambda cond: lambda a: lambda b: a() if cond else b()
def If(cond):
    return IfCond

def IfCond(a):
    return IfCondA

def IfCondA(b):
    return a() if cond else b()

def Max(a,b):
    return If(a>b)(a)(b)

print('Max(3)(5)=', Max(3,5))
