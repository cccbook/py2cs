# 深度學習模型的比較研究 -- 以 MNIST 為例

## 實驗結果 (只跑 3 個 epoch)

```
fc1    92%
fc2    91.97%
fc2relu 94.71%
fc2sig  91.26%
lenet  11%
lenetRelu 95.56%
lenetReluDrop 95.55%
lenetSimplify 96.81%
lenetSimplify2 97.17%
lenetSimplify3 97.81%
lenetSimplify4 97.32%
```

<!--
## Experiments

```
lenet  97%
fc2net 95%
fc2    10%  // overflow error!
fc2s   92%
fc1    74%
fc1s   92%
```
-->

## run under bash

```
$ ./report.sh
```

## Reference

* [卷積神經網絡 CNN 經典模型 — LeNet、AlexNet、VGG、NiN with Pytorch code](https://medium.com/ching-i/%E5%8D%B7%E7%A9%8D%E7%A5%9E%E7%B6%93%E7%B6%B2%E7%B5%A1-cnn-%E7%B6%93%E5%85%B8%E6%A8%A1%E5%9E%8B-lenet-alexnet-vgg-nin-with-pytorch-code-84462d6cf60c)
* [MNIST Handwritten Digit Recognition in PyTorch](https://nextjournal.com/gkoehler/pytorch-mnist)
