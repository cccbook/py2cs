# MonteCarlo

## 執行

```
mac020:montecarlo mac020$ python3 monteCarloPi.py 
MonteCarloPi(100000)= 3.147
mac020:montecarlo mac020$ python3 monteCarloPi.py 
MonteCarloPi(100000)= 3.13804
mac020:montecarlo mac020$ python3 monteCarloPi.py 
MonteCarloPi(100000)= 3.14444
mac020:montecarlo mac020$ python3 monteCarloPi.py 
MonteCarloPi(100000)= 3.13548
```

## 原理

參考圖： [用 monteCarlo 法算 Pi](https://zh.wikipedia.org/wiki/%E8%92%99%E5%9C%B0%E5%8D%A1%E7%BE%85%E6%96%B9%E6%B3%95#/media/File:Pi_30K.gif)

主要想法：

```
四分之一圓的面積 pi/4
正方形面積為 1

落在圓裡面的點 / 所有點 = 四分之一圓的面積 / 正方形面積 = pi/4

pi = 4 * (落在圓裡面的點 / 所有點) = 4 * (hits / n)
```

## 參考文獻

* [R语言与Markov Chain Monte Carlo(MCMC)方法学习笔记(1)] (https://blog.csdn.net/yujunbeta/article/details/21303341) (讚讚，有 R 代碼)
  * 蒙特卡洛方法被誉为20世纪最伟大的十大算法之一。它由美国拉斯阿莫斯国家实验室的三位科学家John von Neumann, Stan Ulam 和 Nick Metropolis于1946年提出。
  * 接受-拒绝抽样（Acceptance-Rejectionsampling): (重要！)
  * Metropolis算法
  * Metropolis-Hastings 算法
  * Bayes推断中的MCMC方法
  * Gibbs抽样
  * [R语言与Markov Chain Monte Carlo(MCMC)方法学习笔记(2)](https://blog.csdn.net/yujunbeta/article/details/23205331)


* [漫談蒙地卡羅法的原理及其應用](https://www.kdais.gov.tw/htmlarea_file/web_articles/kdais/4680/23-1-3.pdf)
  * Pi 的估計！
  * 用中央極限定理作檢定。
  * 高維度積分難以實踐，只好採用蒙地卡羅法估計。

