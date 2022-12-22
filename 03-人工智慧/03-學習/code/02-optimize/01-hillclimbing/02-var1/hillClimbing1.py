# 簡易爬山演算法 -- 針對單變數函數
def hillClimbing(f, x, dx=0.01):
    while (True):
        print('x={0:.5f} f(x)={1:.5f}'.format(x, f(x)))
        if f(x+dx)>f(x): # 如果右邊的高度 f(x+dx) > 目前高度 f(x) ，那麼就往右走
            x = x + dx
        elif f(x-dx)>f(x): # 如果左邊的高度 f(x-dx) > 目前高度 f(x) ，那麼就往左走
            x = x - dx
        else: # 如果兩邊都沒有比現在的 f(x) 高，那麼這裡就是區域最高點，直接中斷傳回
            break
    return x

# 高度函數
def f(x):
    # return -1*(x*x-2*x+1)
    return -1*(x*x+3*x+5)
    # return -1*abs(x*x-4)

hillClimbing(f, 0) # 以 x=0 為起點，開始呼叫爬山演算法
