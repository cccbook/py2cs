# my_mnist

Conv: N,C,H,W
    N is a batch size
    C denotes a number of channels
    H is a height of input planes in pixels
    W is width in pixels


MNIST 的輸入 N=batch_size, C=1, H=28, W=28

經過 nn.Conv2d(1, 10, kernel_size=5) 之後

會得到 N=batch_size, C=10, H=24, W=24

## 卷積的意義

* [【CNN】很詳細的講解什麼以及為什麼是卷積（Convolution）！](https://iter01.com/480243.html)
* [卷積神經網路(Convolutional neural network, CNN) — 卷積運算、池化運算](https://chih-sheng-huang821.medium.com/%E5%8D%B7%E7%A9%8D%E7%A5%9E%E7%B6%93%E7%B6%B2%E8%B7%AF-convolutional-neural-network-cnn-%E5%8D%B7%E7%A9%8D%E9%81%8B%E7%AE%97-%E6%B1%A0%E5%8C%96%E9%81%8B%E7%AE%97-856330c2b703)

## pytorch

* View: 當於 reshape， -1 代表自動設定該維大小，例如： 
    * x 為 `4*2` 的矩陣， x= x.view(2,2,2) 則變成 `2*2*2` 的矩陣
    * x 為 `4*2` 的矩陣， x= x.view(-1,4) 則變成 `2*4` 的矩陣
    * x 為 `4*2` 的矩陣， x= x.view(2,-1,2) 則變成 `2*2*2` 的矩陣

* [PyTorch中view的用法](https://blog.csdn.net/york1996/article/details/81949843)

