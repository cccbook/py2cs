from vfield import *
# 我們想找函數 f 的最低點
def f(p):
    [x,y] = p
    print('x,y=', x, y)
    return x*x + y*y

p = np.array([1.0,3])
print('df(f, p, 0) = ', df(f, p, 0))
print('df(f, p, 1) = ', df(f, p, 1))
print('grad(f)=', grad(f, p))
