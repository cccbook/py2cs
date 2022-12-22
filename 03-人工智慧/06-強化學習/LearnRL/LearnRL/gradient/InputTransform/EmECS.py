## packages
import numpy as np

## define function
def func1(r, xs):
    '''
    f(x) = r * (x,1) / norm2(x,1)
    '''
    # reshape xs
    ListLen = len(xs)
    val = np.array(xs).reshape(ListLen, -1)
    # add ones to xs
    addones = np.ones((ListLen, 1))
    val = np.concatenate((val, addones), axis = 1)
    # norm and 
    norm = np.linalg.norm(val, ord = 2, axis = 1).reshape(ListLen, -1)
    val = (r*val)/norm
    return val

def func2(xs):
    '''
    f(x) = (x, g(x)), g(x) = pow(norm2(x), 2)
    '''
    # reshape xs
    ListLen = len(xs)
    val = np.array(xs).reshape(ListLen, -1)
    # g(x)
    norm = np.linalg.norm(val, ord = 2, axis = 1).reshape(ListLen, -1)
    normPow = pow(norm, 2)
    # append g(x) to xs
    val = np.concatenate((val, normPow), axis = 1)
    return val

def func3(r, xs):
    '''
    f(x) = (f1(x), f2(x))
    '''
    xs1 = func1(r, xs)
    xs2 = func2(xs)
    val = np.concatenate((xs1, xs2), axis = 1)
    return val
    
## main
def main(radius, xs):
    ys1 = func1(radius, xs)
    ys2 = func2(xs)
    ys3 = func3(radius, xs)
    print(ys1)
    print(ys2)
    print(ys3)

if __name__ == '__main__':
    xs = [x for x in range(10)]
    main(3, xs)