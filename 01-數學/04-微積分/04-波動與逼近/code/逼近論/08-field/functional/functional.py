def fadd(f1,f2):
    return lambda x:f1(x)+f2(x)

def fmul(f1,f2):
    return lambda x:f1(x)*f2(x)

f1 = fadd(lambda x:x**2, lambda x:2*x)
print('f1(3)=x^2+2x=', f1(3))

# 函數也可形成 +-*/ 的代數結構，這就是泛函分析的主題。
