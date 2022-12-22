def f1(x):
  return x+1

def f2(x):
  return x*x

def fadd(fa, fb):
  return lambda x:fa(x)+fb(x)

def fmul(fa, fb):
  return lambda x:fa(x)*fb(x)

f = fmul(f1,f2)
print('f(3)=', f(3))
