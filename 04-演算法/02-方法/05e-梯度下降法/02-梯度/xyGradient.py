# 雙變數函數的梯度計算
def grad(f, p, h=0.001):
    x,y = p
    dfx = f([x+h,y])-f([x,y])
    dfy = f([x,y+h])-f([x,y])
    return [dfx/h,dfy/h]

def f(p):
    [x,y] = p
    return x * x + y * y

print('grad(f, [1,1])=', grad(f, [1,2]))