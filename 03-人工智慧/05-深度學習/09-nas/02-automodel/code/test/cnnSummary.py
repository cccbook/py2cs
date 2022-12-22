from torch import nn
from torchsummary import summary

model = nn.Sequential(
          nn.Conv2d(1,6,3),
          nn.ReLU(),
          nn.Conv2d(6,12,3),
          nn.ReLU()
        )
summary(model, (1,28,28))