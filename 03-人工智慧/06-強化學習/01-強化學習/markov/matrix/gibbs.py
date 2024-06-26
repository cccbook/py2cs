# Gibbs Algorithm 的範例
# 問題：機率式有限狀態機，P(a=>b)=0.3, P(b=>a)=0.5 ; P(a=>b)=0.7, P(b=>b)=0.5
# 目標：尋找該「機率式有限狀態機」的穩態，也就是 P(a) = ?, P(b)=? 時系統會達到平衡。
from prob import *

def gibbs (P, T):
    P0 = P.copy()
    print('P0 = {}'.format(P0))
    while True:
        P1 = np.dot(P0, T) # 下一輪的機率分布。
        print('P1={}', P1)
        dp = P1-P0 # 差異
        step = np.linalg.norm(dp) # 差異的大小
        P0 = P1
        if (step < 0.001): break # 假如差異夠小的時候，就可以停止了。

    print('標準答案:P(a)=5/8={} P(b)=3/8={}'.format(5 / 8, 3 / 8)) # 印出標準答案，以便看看我們找到的答案是否夠接近。

gibbs(P, T)
