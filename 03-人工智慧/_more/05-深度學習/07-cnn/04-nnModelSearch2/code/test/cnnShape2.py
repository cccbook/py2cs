import torch
import torch.nn as nn

class Model(nn.Module):
  def __init__(self):
    super(Model, self).__init__()
    self.net = nn.Sequential(
      nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size=5), 
      nn.ReLU(), 
      nn.MaxPool2d(2),
      nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size=5), 
      nn.ReLU(),
      nn.Flatten(),
      nn.Linear(179776, 64),
      nn.ReLU(),
      nn.Linear(64, 10))

  def forward(self, x):
    for layer in self.net:
        x = layer(x)
        print(x.size())
    return x

model = Model()
x = torch.randn(1, 3, 224, 224)

# Let"s print it
model(x)
