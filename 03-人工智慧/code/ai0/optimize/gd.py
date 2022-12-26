from math0 import grad
from numpy.linalg import norm
import numpy as np

# 使用梯度下降法尋找函數最低點
def gd(f, p, step=0.01, max_loops=100000, dump_period=1000):
    for i in range(max_loops):
        fp = f(p)
        gp = grad(f, p) # 計算梯度 gp
        glen = norm(gp) # norm = 梯度的長度 (步伐大小)
        if i%dump_period == 0: 
            print('{:05d}:f(p)={:.3f} p={:s} gp={:s} glen={:.5f}'.format(i, fp, str(p), str(gp), glen))
        if glen < 0.00001: # or fp0 < fp:  # 如果步伐已經很小了，或者 f(p) 變大了，那麼就停止吧！
            break
        gstep = np.multiply(gp, -1*step) # gstep = 逆梯度方向的一小步
        p +=  gstep # 向 gstep 方向走一小步
    print('{:05d}:f(p)={:.3f} p={:s} gp={:s} glen={:.5f}'.format(i, fp, str(p), str(gp), glen))
    return p # 傳回最低點！

if __name__=="__main__":
    def f(p):
        [x, y, z] = p
        return (x-1)**2+(y-2)**2+(z-3)**2

    p = [0.0, 0.0, 0.0]
    gradientDescendent(f, p)

