def f(x):
    # return x*x
    return x**3

h = 0.001

def diff(f, x):
    df = f(x+h)-f(x)
    return df/h

print('diff(f,2)=', diff(f, 2))
