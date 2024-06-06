# Pretrained Model

來源 -- https://www.learnopencv.com/pytorch-for-beginners-image-classification-using-pre-trained-models/ 

參考

1. [PyTorch for Beginners: Image Classification using Pre-trained models](https://learnopencv.com/pytorch-for-beginners-image-classification-using-pre-trained-models/) (讚)
2. [從頭訓練大Model?想多了 : Torchvision 簡介](https://ithelp.ithome.com.tw/articles/10218698)


## alexnet.py

```
$ python alexnet.py
img_t.shape= torch.Size([3, 224, 224])
batch_t.shape= torch.Size([1, 3, 224, 224])
preds.shape= torch.Size([1, 1000])
Labrador retriever
```

## predict.py

```
(env) mac020:pretrained mac020$ python3 predict.py alexnet img/dog.jpg
img_t.shape= torch.Size([3, 224, 224])
batch_t.shape= torch.Size([1, 3, 224, 224])
preds.shape= torch.Size([1, 1000])
Labrador retriever

(env) mac020:pretrained mac020$ python3 predict.py alexnet img/cat.jpg
img_t.shape= torch.Size([3, 224, 224])
batch_t.shape= torch.Size([1, 3, 224, 224])
preds.shape= torch.Size([1, 1000])
Egyptian cat
```

windows

```
PS D:\pmedia\陳鍾誠\課程\人工智慧\08-deep\02-pretrained\01-torchvision\01-classify> python predict.py alexnet img/bee.jpg
nts\alexnet-owt-4df8aa71.pth
100%|███████████████████████████████████████| 233M/233M [02:40<00:00, 1.53MB/s]
img_t.shape= torch.Size([3, 224, 224])
batch_t.shape= torch.Size([1, 3, 224, 224])
preds.shape= torch.Size([1, 1000])
harvestman, daddy longlegs, Phalangium opilio
PS D:\pmedia\陳鍾誠\課程\人工智慧\08-deep\02-pretrained\01-torchvision\01-classify> python predict.py alexnet img/dog.jpg
img_t.shape= torch.Size([3, 224, 224])
batch_t.shape= torch.Size([1, 3, 224, 224])
preds.shape= torch.Size([1, 1000])
Labrador retriever
PS D:\pmedia\陳鍾誠\課程\人工智慧\08-deep\02-pretrained\01-torchvision\01-classify> python predict.py alexnet img/cat.jpg
img_t.shape= torch.Size([3, 224, 224])
batch_t.shape= torch.Size([1, 3, 224, 224])
preds.shape= torch.Size([1, 1000])
Egyptian cat
```