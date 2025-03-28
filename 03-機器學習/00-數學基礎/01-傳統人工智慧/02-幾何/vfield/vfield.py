import numpy as np

pi = np.pi
step = 0.001

# 函數 f 對變數 p[k] 的偏微分: df / dp[k]
def df(f, p, k):
    p1 = p.copy()
    p1[k] = p[k]+step
    return (f(p1) - f(p)) / step

# 單變數微分
def df1(f, x):
    return df(f, np.array([x]), 0)

# 梯度：函數 f 在點 p 上的梯度
def grad(f, p):
    gp = p.copy()
    for k in range(len(p)):
        gp[k] = df(f, p, k)
    return gp

# 散度：函數 f 在點 p 上的散度
# 限制：f 為 n 維向量函數 f(Rn)->Rn
def div(f, p):
    r = 0
    for i in range(p.size):
        r += df(f, p, i)[i]
    return r

# 旋度：函數 f 在點 p 上的旋度
# 限制：f 為 3 維向量函數 f(R3)->R3
def curl(f, p):
    rx = df(f, p, 1)[2]-df(f, p, 2)[1]
    ry = df(f, p, 2)[0]-df(f, p, 0)[2]
    rz = df(f, p, 0)[1]-df(f, p, 1)[0]
    return np.array([rx, ry, rz])

# 拉普拉斯算子 ∆ = div∇
def laplace(f, p):
    return div(lambda p:grad(f, p), p)

def lintegrate(F, r, a, b, dt=step):
    area = 0.
    t = a
    while (t < b):
        p = r(t)
        dr = df1(r, t)
        # print('t=: p=: F(p)=:'.format(t, p, F(p)))
        area += np.dot(F(p), dr)*dt
        t += dt
    return area



