import numpy as np

step = 0.01

# 我們想找函數 f 的最低點
def f(p):
    [x,y] = p
    # print('x,y=', x, y)
    return x*x + y*y

# # 函數 f 對變數 p[k] 的偏微分: df / dp[k]
def df(f, p, k):
    p1 = p.copy()
    p1[k] = p[k]+step
    return (f(p1) - f(p)) / step

# 函數 f 在點 p 上的梯度
def grad(f, p):
    gp = p.copy()
    for k in range(len(p)):
        gp[k] = df(f, p, k)
    return gp

p = np.array([1.0,3.0])
print('df(f, p, 0) = ', df(f, p, 0))
print('df(f, p, 1) = ', df(f, p, 1))
print('grad(f)=', grad(f, p))
