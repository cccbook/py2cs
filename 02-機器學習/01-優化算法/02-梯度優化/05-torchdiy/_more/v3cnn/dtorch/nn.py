import torch
import torch.nn as nn

from . import linear
from . import activate
from . import loss
from . import cnn
from . import dropout

Module = nn.Module
Dropout = dropout.Dropout # nn.Dropout

Linear = linear.Linear
ReLU = activate.ReLU
MaxPool2d = cnn.MaxPool2d
Conv2d = nn.Conv2d # 速度的問題，使用 Conv2d = cnn.Conv2d 會變得很慢，所以還是維持用 nn.Conv2d
CrossEntropyLoss = loss.CrossEntropyLoss
