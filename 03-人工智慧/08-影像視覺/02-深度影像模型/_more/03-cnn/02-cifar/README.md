# cifar10 -- 10 種影像物件辨識

## 參考

* https://ithelp.ithome.com.tw/articles/10218698
* https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

## predict

```
mac020:cifar mac020$ python3 cifar.py predict
device= cpu
Files already downloaded and verified
Files already downloaded and verified
Answer:  bird  frog   cat truck
Predicted:   bird  frog   cat truck
```

## test

```
mac020:cifar mac020$ python3 cifar.py test
device= cpu
Files already downloaded and verified
Files already downloaded and verified
Accuracy of the network on the 10000 test images: 58 %
```

## train

```
mac020:cifar mac020$ python3 cifar.py train
device= cpu
Files already downloaded and verified
Files already downloaded and verified
[1,  2000] loss: 2.207
[1,  4000] loss: 1.865
[1,  6000] loss: 1.668
[1,  8000] loss: 1.590
[1, 10000] loss: 1.509
[1, 12000] loss: 1.470
[2,  2000] loss: 1.416
[2,  4000] loss: 1.363
[2,  6000] loss: 1.353
[2,  8000] loss: 1.332
[2, 10000] loss: 1.320
[2, 12000] loss: 1.282
[3,  2000] loss: 1.213
[3,  4000] loss: 1.206
[3,  6000] loss: 1.215
[3,  8000] loss: 1.211
[3, 10000] loss: 1.211
[3, 12000] loss: 1.186
Finished Training
```

