import numpy as np
import math
# 參考文獻：Numerical example to understand Expectation-Maximization -- http://ai.stanford.edu/~chuongdo/papers/em_tutorial.pdf
# What is the expectation maximization algorithm? (PDF) -- http://stats.stackexchange.com/questions/72774/numerical-example-to-understand-expectation-maximization

'''
計算 P(e|p)

ex: logP(e1|pA) = logP([5,5]|[0.6,0.4]) 
               = log(0.6^5*0.4^5) 
               = 5 log(0.6) + 5 log(0.4)

最後 P(e1|pA) = exp(logP(e1|pA)) = 0.6^5 + 0.4^5
'''
def P(e, p):
    logP = np.dot(e, np.log(p))
    return np.exp(logP)

'''
1st:  Coin B, {HTTTHHTHTH}, 5H,5T
2nd:  Coin A, {HHHHTHHHHH}, 9H,1T
3rd:  Coin A, {HTHHHHHTHH}, 8H,2T
4th:  Coin B, {HTHTTTHHTT}, 4H,6T
5th:  Coin A, {THHHTHHHTH}, 7H,3T
假如已經知道，1st, 4th 是 B, 2nd, 3rd, 5th 是 A, 
那就可以計算出 pA(heads) = 0.80 and pB(heads)=0.45
'''
def EM():
    e = [ [5,5], [9,1], [8,2], [4,6], [7,3] ]
    pA = [0.6, 0.4]
    pB = [0.5, 0.5]
    delta = 9.9999
    for _ in range(1000):
        print("pA={} pB={} delta={}".format(pA, pB, delta))
        sumA=[0,0]
        sumB=[0,0]
        for ei in e:
            # estimate
            a = P(ei, pA) # 用 A 去擲會產生 ei 的機率之 log
            b = P(ei, pB)
            wA = a/(a+b)   # = a 的權重
            wB = b/(a+b)   # = b 的權重
            eA = np.multiply(wA, ei) # A 的估計值
            eB = np.multiply(wB, ei) # B 的估計值
            # estimation 完成估計
            # maximization （先加總）
            sumA = np.add(sumA, eA)
            sumB = np.add(sumB, eB)

        npA = np.multiply(sumA, 1.0/np.sum(sumA)) # 新一版的 pA
        npB = np.multiply(sumB, 1.0/np.sum(sumB)) # 新一版的 pB
        # 計算差異，看看是否該停止了
        dA  = np.subtract(npA, pA)
        dB  = np.subtract(npB, pB)
        delta = np.max([dA, dB])
        if delta < 0.001: break
        # 更新 pA, pB 為新一版
        pA = npA
        pB = npB

EM()
