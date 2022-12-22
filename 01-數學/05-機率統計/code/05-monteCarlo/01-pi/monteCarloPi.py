'''
參考圖： https://zh.wikipedia.org/wiki/%E8%92%99%E5%9C%B0%E5%8D%A1%E7%BE%85%E6%96%B9%E6%B3%95#/media/File:Pi_30K.gif

四分之一圓的面積 pi/4
正方形面積為 1

落在圓裡面的點 / 所有點 = 四分之一圓的面積 / 正方形面積 = pi/4

pi = 4 * (落在圓裡面的點 / 所有點) = 4 * (hits / n)
'''
from random import random

def monteCarloPi(n):
    hits = 0
    for _ in range(n):
        x = random()
        y = random()
        if (x*x+y*y <= 1):
            hits += 1
    return 4*(hits/n)

print('MonteCarloPi(100000)=', monteCarloPi(100000))
