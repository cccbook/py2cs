## EM 演算法實作

看過 EM 演算法的數學介紹之後，相信大家應該和我一樣，都是一頭霧水的。誰知道那些數學要怎麼寫成程式呢？

這個問題困擾了我很久，直到我看到下列的這則討論和論文中的那個範例，我才開始真正能體會那些數學的用法。

* [Numerical example to understand Expectation-Maximization](http://stats.stackexchange.com/questions/72774/numerical-example-to-understand-expectation-maximization) 
* [What is the expectation maximization algorithm? (PDF)](http://ai.stanford.edu/~chuongdo/papers/em_tutorial.pdf) 

一個好的範例，果然勝過千言萬語！

### 範例

這個範例探討的是兩個「不公正的銅板」 A 與 B，兩者正面的機率分別是 $`\theta A`$ 與 $`\theta B`$ ，當我們用這兩的銅板進行一系列的抽樣時，得到了下列的結果。

![圖、本圖來自 What is the expectation maximization algorithm? 一文](./img/em1.png)

上圖中的 H 代表正面 (Head) ，而 T 代表反面 (Tail)。

由於我們知道每個實驗是由哪個銅板造成的，因此就可以直接用「點估計」的方式，使用「樣本平均值」 $`\hat{\theta}_A`$ 與 $`\hat{\theta}_B`$ 作為 $`\theta_A`$ 與 $`\theta_B`$ 兩者的「不偏估計量」。

但是、假如今天我們不知道每個序列到底是由哪個銅板所產生的，那麼我們應該怎麼樣估計 $`\hat{\theta}_A`$ 與 $`\hat{\theta}_B`$ 呢？

這個問題就比較困難了！

還好、我們有 EM 演算法可以幫忙，這正是 EM 演算法神奇的地方。下圖顯示了 EM 演算法在這個範例上的運作過程。

![圖、本圖來自 What is the expectation maximization algorithm? 一文](./img/em2.png)

首先、 EM 演算法先隨便設 $`(\hat{\theta}_A, \hat{\theta}_B)`$ 兩個參數的初值為 $`(0.6, 0.5)`$ (如圖中 1 的部分)，然後開始進行 Expectation (期望 E 程序) 與 Maximization (最大化 M 程序) 的循環。

在 E 程序中 (如圖中 2 的部分)，EM 演算法透過這組 $`(\hat{\theta}_A=0.6, \hat{\theta}_B=0.5)`$ 計算出在這些樣本序列中，每個序列來自 A 或 B 的機率。例如在第一組樣本中，來自 A 的機率為 0.45，而來自 B 的機率為 0.55。

(註：這裏的 0.45 的來源是 $`\frac{C(10,5)*0.6^5*0.4^5}{C(10,5)*(0.6^5*0.4^5+0.5^5*0.5^5)}=0.45`$ ，而 0.55 則是 1-0.45 ，至於其數學分析請看後文)。

接著、再用這個機率去計算出銅板「正」(H) 「反」(T) 兩面出現次數的期望值。舉例而言， P(A) * (#H, #T) = 0.45 * (5H, 5T) = (0.225 H, 0.225T) ~ (2.2H, 2.2T) 。

同樣的，P(B)*(#H, #T) = 0.55 * (5H,5T) = (2.75 H, 2.75T) ~ (2.8H, 2.8T) 。

當然，後面序列的正反面期望值也都是比照這種作法計算出來的。

然後、就可以進行 M 程序，計算「最大可能的 $`(\hat{\theta}_A, \hat{\theta}_B)`$ 值」，結果得到 $`\hat{\theta}_A=\frac{21.3}{21.3+8.6}=0.71, \hat{\theta}_B=\frac{11.7}{11.7+8.4}=0.58`$ 。

如此就能進行完一輪「E-M 程序」，讓 $`(\hat{\theta}_A, \hat{\theta}_B)`$ 的估計變得更好。

於是、只要我們反覆的進行 E-M 程序，最後 $`(\hat{\theta}_A, \hat{\theta}_B)`$ 就會收斂到某個值上，此時就代表訓練完成了。在上圖中、最後 $`(\hat{\theta}_A = 0.80, \hat{\theta}_B = 0.52)`$ 

### EM 演算法的數學原理

上述 EM 演算法問題的背後，其實是一種「最大概似估計」，只是加入了「隱變數」的概念，這種「最大概似估計」企圖最大化下列算式中的 $`\theta`$ 值。

```math
L(X;\theta)=P(X|\theta)=\sum_z P(X,z|\theta)
```

但問題是、假如 z 是個序列 (`z[1..n]`)，那麼 z 的所有可能性將會非常的大 (指數膨脹)，因此要計算上述總和會變得非常困難且耗時，因此我們通常無法直接計算上述的「總合算式」。

但是、我們已經有一些 X 的樣本了，假如我們用「這些樣本的統計值」(x1, x2, ...., xn) 來當作所有可能的 z 總合的一個估計式，也就是用 $`\sum_i P(x_i|\theta)/n`$ 來估計 $`\sum_z P(X,z|\theta) P(X,z|\theta)`$ ，那就可以避開「指數膨脹」的問題，快速的計算出 $`P(X|\theta)`$ 的結果。

```math
L(X;\theta)=P(X|\theta)=\sum_z P(X,z|\theta) \simeq \sum_i P(x_i|\theta)/n
```

上述算式，就是所謂的 E 程序 (估計 Expectation)。

然後、我們就可以在下一回合的 M 程序中，找出下列算式的優化結果。

```math
\arg\max_{\theta_{t+1}} \sum_i P(x_i|\theta)/n
```

### 兩銅板範例的數學分析

在上述的兩銅板範例中， $`X, Z, \theta`$ 參數的對應如下：

> $`\theta=(\theta_A,\theta_B)`$ ;
> $`z=(z_1,z_2,z_3,z_4,z_5)`$ ; 其中的 $`z_i`$ 代表第 i 次實驗到底是由 A 還是 B 產生的。
> 
> X 是一個隨機變數，將十次的投擲映射到正面次數，例如： $`X(HTTTHHTHTH)=5`$ ; 

但是、在上述問題中，其實有兩個不同的分布，也就是 A 與 B，因此當我們要計算 $`P(x_i|\theta)=P(x_i|\theta_A,\theta_B)`$ 時，我們可以分配給 A, B 兩者一定的權重 $`(W_A, W_B)`$ ，只要這兩個權重相加為 1 既可。

但是 $`(W_A, W_B)`$ 應該各分配多少權重呢？一個想法是採用條件機率的方法如下：

> 讓 $`W_A=P(x_i|\theta_A);W_B=P(x_i|\theta_B)`$ 。
> 
> 於是、對於 $`x1=(5H,5T)`$ 而言，我們就可以用 $`W_A=\frac{C(10,5)*0.6^5*0.4^5}{C(10,5)*(0.6^5*0.4^5+0.5^5*0.5^5)}=0.45`$ 的比例進行分配。 
> 
> 然後設定 $`W_B=1-W_A=1-0.45=0.55`$ 。 

### 程式實作
以下是針對上述範例的程式實作，整個程式含註解只有 39 行。

其中的 pA, pB 就是公式中的 $`\theta A ; \theta B`$ 參數， e 基本上就是 Ｘ，而 zi 所代表的第 i 次實驗到底是由 A 還是 B 產生的，則是由 wA, wB 的期望值完成的。

程式： EM.py

```py
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

```

### 執行結果

以下是上述程式的執行結果，您可以看到該結果和圖中範例的數據是吻合的。

```
mac020:em mac020$ python3 em.py
pA=[0.6, 0.4] pB=[0.5, 0.5] delta=9.9999
pA=[0.71301224 0.28698776] pB=[0.58133931 0.41866069] delta=0.11301223540051619
pA=[0.74529204 0.25470796] pB=[0.56925575 0.43074425] delta=0.0322798006814784
pA=[0.76809883 0.23190117] pB=[0.54953591 0.45046409] delta=0.022806798285326613
pA=[0.78316458 0.21683542] pB=[0.53461745 0.46538255] delta=0.015065749932652417
pA=[0.79105525 0.20894475] pB=[0.52628117 0.47371883] delta=0.008336287117588381
pA=[0.79453254 0.20546746] pB=[0.52239044 0.47760956] delta=0.003890729512057156
pA=[0.79592867 0.20407133] pB=[0.52072988 0.47927012] delta=0.001660559431849007
```

### 參考文獻
* [Numerical example to understand Expectation-Maximization](http://stats.stackexchange.com/questions/72774/numerical-example-to-understand-expectation-maximization)
* [how does expectation maximization work?](http://math.stackexchange.com/questions/25111/how-does-expectation-maximization-work)
* [how does expectation maximization work in coin flipping problem](http://math.stackexchange.com/questions/81004/how-does-expectation-maximization-work-in-coin-flipping-problem)
* [What is the expectation maximization algorithm? (PDF)](http://ai.stanford.edu/~chuongdo/papers/em_tutorial.pdf) (讚！)
