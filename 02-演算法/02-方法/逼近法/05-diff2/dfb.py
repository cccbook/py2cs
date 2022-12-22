def df1(f, x, h=0.00001):
    return (-f(x+2*h)+8*f(x+h)-8*f(x-h)+f(x-2*h))/(12*h)

def df2(f, x, h=0.00001):
    return (-f(x+2*h)+16*f(x+h)-30*f(x)+16*f(x-h)-f(x-2*h))/(12*(h**2))

def df3(f, x, h=0.00001):
    return (-f(x+3*h)+8*f(x+2*h)-13*f(x+h)+13*f(x-h)-8*f(x-2*h)+f(x-3*h))/(8*(h**3))

def df4(f, x, h=0.00001):
    return (-f(x+3*h)+12*f(x+2*h)-39*f(x+h)+56*f(x)-39*f(x-h)+12*f(x-2*h)+f(x-3*h))/(6*(h**4))
