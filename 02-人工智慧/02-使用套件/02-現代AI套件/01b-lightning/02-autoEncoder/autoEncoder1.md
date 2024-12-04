

```
% python autoEncoder1.py
Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
Failed to download (trying next):
<urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: certificate has expired (_ssl.c:1000)>

Downloading https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz
Downloading https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz to ./MNIST/raw/train-images-idx3-ubyte.gz
100%|█████████████| 9912422/9912422 [00:09<00:00, 1029920.65it/s]
Extracting ./MNIST/raw/train-images-idx3-ubyte.gz to ./MNIST/raw

Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
Failed to download (trying next):
<urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: certificate has expired (_ssl.c:1000)>

Downloading https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz
Downloading https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz to ./MNIST/raw/train-labels-idx1-ubyte.gz
100%|██████████████████| 28881/28881 [00:00<00:00, 113195.91it/s]
Extracting ./MNIST/raw/train-labels-idx1-ubyte.gz to ./MNIST/raw

Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
Failed to download (trying next):
<urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: certificate has expired (_ssl.c:1000)>

Downloading https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz
Downloading https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz to ./MNIST/raw/t10k-images-idx3-ubyte.gz
100%|██████████████| 1648877/1648877 [00:07<00:00, 223157.22it/s]
Extracting ./MNIST/raw/t10k-images-idx3-ubyte.gz to ./MNIST/raw

Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
Failed to download (trying next):
<urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: certificate has expired (_ssl.c:1000)>

Downloading https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz
Downloading https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz to ./MNIST/raw/t10k-labels-idx1-ubyte.gz
100%|███████████████████| 4542/4542 [00:00<00:00, 5395221.97it/s]
Extracting ./MNIST/raw/t10k-labels-idx1-ubyte.gz to ./MNIST/raw

GPU available: True (mps), used: True
TPU available: False, using: 0 TPU cores
HPU available: False, using: 0 HPUs
/opt/miniconda3/lib/python3.12/site-packages/lightning/pytorch/loops/utilities.py:72: `max_epochs` was not set. Setting it to 1000 epochs. To train without an epoch limit, set `max_epochs=-1`.
/opt/miniconda3/lib/python3.12/site-packages/lightning/pytorch/trainer/configuration_validator.py:68: You passed in a `val_dataloader` but have no `validation_step`. Skipping val loop.

  | Name    | Type       | Params | Mode 
-----------------------------------------------
0 | encoder | Sequential | 100 K  | train
1 | decoder | Sequential | 101 K  | train
-----------------------------------------------
202 K     Trainable params
0         Non-trainable params
202 K     Total params
0.810     Total estimated model params size (MB)
8         Modules in train mode
0         Modules in eval mode
/opt/miniconda3/lib/python3.12/site-packages/lightning/pytorch/trainer/connectors/data_connector.py:424: The 'train_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=7` in the `DataLoader` to improve performance.
Epoch 0:  53%|██  | 29034/55000 [01:23<0Epoch 0:  53%|██  | 29034/55000 [01:23<0Epoch 0:  53%|██  | 29035/55000 [01:23<0Epoch 0:  53%|██  | 29035/55000 [01:23<0Epoch 0:  53%|██  | 29036/55000 [01:23<0Epoch 0:  53%|██  | 29036/55000 [01:23<0Epoch 0:  53%|█▌ | 29037/55000 [01:23<01Epoch 0:  53%|█▌ | 29038/55000 [01:23<01Epoch 0:  53%|█▌ | 29038/55000 [01:23<01Epoch 0:  53%|█ | 29039/55000 [01:23<01:Epoch 0:  53%|█ | 29039/55000 [01:23<01:Epoch 0:  53%|█ | 29040/55000 [01:23<01:Epoch 0:  53%|█ | 29040/55000 [01:23<01:Epoch 0:  53%|█ | 29041/55000 [01:23<01:Epoch 0:  53%|█ | 29041/55000 [01:23<01:Epoch 0:  53%|█ | 29042/55000 [01:23<01:Epoch 0:  53%|█ | 29042/55000 [01:23<01:Epoch 0:  53%|█ | 29043/55000 [01:23<01:Epoch 0:  53%|█ | 29043/55000 [01:23<01:Epoch 0:  53%|▌| 29044/55000 [01:23<01:1Epoch 0:  53%|▌| 29044/55000 [01:23<01:1Epoch 0:  53%|▌| 29045/55000 [01:23<01:1Epoch 0:  53%|▌| 29045/55000 [01:23<01:1Epoch 0:  53%|▌| 29046/55000 [01:23<01:1Epoch 0:  53%|▌| 29046/55000 [01:23<01:1Epoch 0:  53%|▌| 29047/55000 [01:23<01:1Epoch 0:  53%|▌| 29047/55000 [01:23<01:1Epoch 0:  53%|▌| 29048/55000 [01:23<01:1Epoch 0:  53%|▌| 29048/55000 [01:23<01:1Epoch 0:  53%|▌| 29049/55000 [01:23<01:1Epoch 0:  53%|▌| 29049/55000 [01:23<01:1Epoch 0:  53%|▌| 29050/55000 [01:23<01:1Epoch 0:  53%|▌| 29050/55000 [01:23<01:1Epoch 0:  53%|▌| 29051/55000 [01:23<01:1Epoch 0:  53%|▌| 29051/55000 [01:23<01:1Epoch 0:  53%|▌| 29052/55000 [01:23<01:1Epoch 0:  53%|▌| 29052/55000 [01:23<01:1Epoch 0:  53%|▌| 29053/55000 [01:23<01:1Epoch 0:  53%|▌| 29053/55000 [01:23<01:1Epoch 0:  53%|▌| 29054/55000 [01:23<01:1Epoch 0:  53%|▌| 29054/55000 [01:23<01:1Epoch 0:  53%|▌| 29055/55000 [01:23<01:1Epoch 0:  53%|▌| 29055/55000 [01:23<01:1Epoch 0:  53%|▌| 29056/55000 [01:23<01:1Epoch 0:  53%|▌| 29056/55000 [01:23<01:1Epoch 0:  53%|▌| 29057/55000 [01:23<01:1Epoch 0:  53%|▌| 29057/55000 [01:23<01:1Epoch 0:  53%|▌| 29058/55000 [01:23<01:1Epoch 0:  53%|▌| 29058/55000 [01:23<01:1Epoch 0:  53%|▌| 29059/55000 [01:23<01:1Epoch 0:  53%|▌| 29059/55000 [01:23<01:1Epoch 0:  53%|▌| 29060/55000 [01:23<01:1Epoch 0:  53%|▌| 29060/55000 [01:23<01:1Epoch 0:  53%|▌| 29061/55000 [01:23<01:1Epoch 0:  53%|▌| 29061/55000 [01:23<01:1Epoch 0:  53%|▌| 29062/55000 [01:23<01:1Epoch 0:  53%|▌| 29062/55000 [01:23<01:1Epoch 0:  53%|▌| 29063/55000 [01:23<01:1Epoch 0:  53%|▌| 29063/55000 [01:23<01:1Epoch 0:  53%|▌| 29064/55000 [01:23<01:1Epoch 0:  53%|▌| 29064/55000 [01:23<01:1Epoch 0:  53%|▌| 29065/55000 [01:23<01:1Epoch 0:  53%|▌| 29065/55000 [01:23<01:1Epoch 0:  53%|▌| 29066/55000 [01:23<01:1Epoch 0:  53%|▌| 29066/55000 [01:23<01:1Epoch 0:  53%|▌| 29067/55000 [01:23<01:1Epoch 0:  53%|▌| 29067/55000 [01:23<01:1Epoch 0:  53%|▌| 29068/55000 [01:23<01:1Epoch 0:  53%|▌| 29068/55000 [01:23<01:1Epoch 0:  53%|▌| 29069/55000 [01:23<01:1Epoch 0:  53%|▌| 29069/55000 [01:23<01:1Epoch 0:  53%|▌| 29070/55000 [01:23<01:1Epoch 0:  53%|▌| 29070/55000 [01:23<01:1Epoch 0:  53%|▌| 29071/55000 [01:23<01:1Epoch 0:  53%|▌| 29071/55000 [01:23<01:1Epoch 0:  53%|▌| 29072/55000 [01:23<01:1Epoch 0:  53%|▌| 29072/55000 [01:23<01:1Epoch 0:  53%|▌| 29073/55000 [01:23<01:1Epoch 0:  53%|▌| 29073/55000 [01:23<01:1Epoch 0:  53%|▌| 29074/55000 [01:23<01:1Epoch 0:  53%|▌| 29074/55000 [01:23<01:1Epoch 0:  53%|▌| 29075/55000 [01:23<01:1Epoch 0:  53%|▌| 29075/55000 [01:23<01:1Epoch 0:  53%|▌| 29076/55000 [01:23<01:1Epoch 0:  53%|▌| 29105/55000 [01:23<01:1Epoch 0:  53%|▌| 29105/55000 [01:23<01:1Epoch 1:  79%|▊| 43525/55000 [02:08<00:33, 339.19it/s, v_nu^C
Detected KeyboardInterrupt, attempting graceful shutdown ...
```