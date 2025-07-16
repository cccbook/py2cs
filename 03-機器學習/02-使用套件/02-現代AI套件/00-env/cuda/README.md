# cuda


## 安裝請選對版本

```
$ pip3 install torch==1.10.0+cu102 torchvision==0.11.1+cu102 torchaudio===0.10.0+cu102 -f https://download.pytorch.org/whl/cu102/torch_stable.html
```

## 測試

```
$ python testcuda.py
torch.cuda.is_available()= True
```

## 03-cnn/02-cifar

錯誤原因應該是我的顯卡太舊，目前的 pytorch 已經不支援了

請參考 [pytorch 报错 RuntimeError: CUDA error: no kernel image is available for execution on the device](https://blog.csdn.net/weixin_42642296/article/details/115598760) 

```
$ python cifar.py
device= cuda:0
Downloading https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz to ./data\cifar-10-python.tar.gz
170499072it [02:08, 1325362.49it/s]
Extracting ./data\cifar-10-python.tar.gz to ./data
Files already downloaded and verified
Traceback (most recent call last):
  File "C:\ccc\course\ai\08-deep\03-cnn\02-cifar\cifar.py", line 137, in <module>
    op = sys.argv[1]
IndexError: list index out of range
$ python cifar.py train
device= cuda:0
Files already downloaded and verified
Files already downloaded and verified
C:\Users\Hero3C\AppData\Local\Programs\Python\Python39\lib\site-packages\torch\cuda\__init__.py:120: UserWarning:
    Found GPU%d %s which is of cuda capability %d.%d.
    PyTorch no longer supports this GPU because it is too old.
    The minimum cuda capability supported by this library is %d.%d.

  warnings.warn(old_gpu_warn.format(d, name, major, minor, min_arch // 10, min_arch % 10))
Traceback (most recent call last):
  File "C:\ccc\course\ai\08-deep\03-cnn\02-cifar\cifar.py", line 141, in <module>
    train()
  File "C:\ccc\course\ai\08-deep\03-cnn\02-cifar\cifar.py", line 73, in train
    outputs = net(inputs)
  File "C:\Users\Hero3C\AppData\Local\Programs\Python\Python39\lib\site-packages\torch\nn\modules\module.py", line 1102, in 
_call_impl
    return forward_call(*input, **kwargs)
  File "C:\ccc\course\ai\08-deep\03-cnn\02-cifar\cifar.py", line 36, in forward
    x = self.pool(F.relu(self.conv1(x)))
  File "C:\Users\Hero3C\AppData\Local\Programs\Python\Python39\lib\site-packages\torch\nn\modules\module.py", line 1102, in 
_call_impl
    return forward_call(*input, **kwargs)
  File "C:\Users\Hero3C\AppData\Local\Programs\Python\Python39\lib\site-packages\torch\nn\modules\conv.py", line 446, in forward
    return self._conv_forward(input, self.weight, self.bias)  
  File "C:\Users\Hero3C\AppData\Local\Programs\Python\Python39\lib\site-packages\torch\nn\modules\conv.py", line 442, in _conv_forward
    return F.conv2d(input, weight, bias, self.stride,
RuntimeError: CUDA error: no kernel image is available for execution on the device
CUDA kernel errors might be asynchronously reported at some other API call,so the stacktrace below might be incorrect.      
For debugging consider passing CUDA_LAUNCH_BLOCKING=1. 
```