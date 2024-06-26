# Logistics Regression

* https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/01-basics/logistic_regression/main.py#L33-L34

```
C:\Users\user\AppData\Local\Programs\Python\Python38\lib\site-packages\torchvision\datasets\mnist.py:502: UserWarning: The given NumPy array is not writeable, and PyTorch does not support non-writeable tensors. This means you can write to the underlying (supposedly non-writeable) NumPy array using the tensor. You may want to copy the array to protect its data or make it writeable before converting it to a tensor. This type of warning will be suppressed for the rest of this program. (Triggered internally at 
 ..\torch\csrc\utils\tensor_numpy.cpp:143.)
  return torch.from_numpy(parsed.astype(m[2], copy=False)).view(*s)
Done!
Epoch [1/5], Step [100/600], Loss: 2.2450
Epoch [1/5], Step [200/600], Loss: 2.1503
Epoch [1/5], Step [300/600], Loss: 2.0727
Epoch [1/5], Step [400/600], Loss: 1.9882
Epoch [1/5], Step [500/600], Loss: 1.9735
Epoch [1/5], Step [600/600], Loss: 1.8059
Epoch [2/5], Step [100/600], Loss: 1.7796
Epoch [2/5], Step [200/600], Loss: 1.6676
Epoch [2/5], Step [300/600], Loss: 1.6033
Epoch [2/5], Step [400/600], Loss: 1.6103
Epoch [2/5], Step [500/600], Loss: 1.5206
Epoch [2/5], Step [600/600], Loss: 1.4200
Epoch [3/5], Step [100/600], Loss: 1.3904
Epoch [3/5], Step [200/600], Loss: 1.3999
Epoch [3/5], Step [300/600], Loss: 1.3513
Epoch [3/5], Step [400/600], Loss: 1.3183
Epoch [3/5], Step [500/600], Loss: 1.2844
Epoch [3/5], Step [600/600], Loss: 1.2350
Epoch [4/5], Step [100/600], Loss: 1.2213
Epoch [4/5], Step [200/600], Loss: 1.1902
Epoch [4/5], Step [300/600], Loss: 1.1247
Epoch [4/5], Step [400/600], Loss: 1.2494
Epoch [4/5], Step [500/600], Loss: 1.1886
Epoch [4/5], Step [600/600], Loss: 1.0922
Epoch [5/5], Step [100/600], Loss: 1.0672
Epoch [5/5], Step [200/600], Loss: 1.0719
Epoch [5/5], Step [300/600], Loss: 1.1387
Epoch [5/5], Step [400/600], Loss: 0.9736
Epoch [5/5], Step [500/600], Loss: 0.9852
Epoch [5/5], Step [600/600], Loss: 1.0538
Accuracy of the model on the 10000 test images: 82.94000244140625 %
```