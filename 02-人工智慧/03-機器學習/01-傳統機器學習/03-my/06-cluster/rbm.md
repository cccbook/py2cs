# RBM -- Restricted Boltzmamn Machine

想法：有很多輸入 V1, V2, ....，我們想建立一個 RBM 讓輸入 Vi 後，能透過隱層與權重，重建出 Vi' ，其中 Vi' 與 Vi 非常接近。

顯層 V 與隱層 H 兩者的重建公式：

$$
P(v|h) = \prod_{i=1}^m P(v_i|h).
$$

$$
P(h|v) = \prod_{j=1}^n P(h_j|v).
$$

如此，只要 權重 W, a, b 都學好了，就能透過上述公式重建 Vi' 了。

RBM 的 CD-1 學習算法

![](https://img-blog.csdn.net/20170407102443967?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvemhpaHVhX29iYQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

簡易算法 -- https://blog.51cto.com/34526667/1893886

* https://pypi.org/project/pydbm/ (讚)
    * 專案 pypi 頁裡面有詳細的原理解說
    * https://code.accel-brain.com/Deep-Learning-by-means-of-Design-Pattern/

* [受限玻爾茲曼機](https://zh.wikipedia.org/zh-tw/%E5%8F%97%E9%99%90%E7%8E%BB%E5%B0%94%E5%85%B9%E6%9B%BC%E6%9C%BA)
    * [訓練算法](https://zh.wikipedia.org/zh-tw/%E5%8F%97%E9%99%90%E7%8E%BB%E5%B0%94%E5%85%B9%E6%9B%BC%E6%9C%BA#%E8%AE%AD%E7%BB%83%E7%AE%97%E6%B3%95)
* [Introduction to Restricted Boltzmann Machines](http://blog.echen.me/2011/07/18/introduction-to-restricted-boltzmann-machines/)
* [受限玻尔兹曼机(RBM)+对比散度算法(CD-k)](https://blog.csdn.net/zhihua_oba/article/details/69487730)
* [對比散度演算法（CD演算法）](https://www.796t.com/content/1546875214.html)

* https://github.com/lmjohns3/rbm
    * [Rectified Linear Units Improve Restricted Boltzmann Machines (PDF)](http://www.csri.utoronto.ca/~hinton/absps/reluICML.pdf)
* https://github.com/echen/restricted-boltzmann-machines/blob/master/rbm.py
* https://gist.github.com/yusugomori/4428308
* https://github.com/basavin/rbm-smple
* https://github.com/ramarlina/RBM/blob/master/rbm.py (讚)
* https://github.com/FengZiYjun/Restricted-Boltzmann-Machine

範例來源

* https://github.com/echen/restricted-boltzmann-machines/blob/master/rbm.py