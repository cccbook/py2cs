# If  = lambda cond: lambda a: lambda b: a() if cond else b()
def If(cond):
    return lambda a:a() if cond else lambda b:b()

def Max(a):
    return lambda b:If(a>b)(lambda:a)(lambda:b)

print('Max(3)(5)=', Max(3)(5))
