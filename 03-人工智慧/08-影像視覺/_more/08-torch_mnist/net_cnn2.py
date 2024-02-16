import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5) # 維度: 輸入 28*28, kernel_size=5, 28-5+1=24 ，所以變成 24*24 的平面
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5) # 維度: 輸入 12*12, kernel_size=5, 12-5+1=8 ，所以變成 8*8 的平面
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2)) # 維度: 1*(28*28)=>10*(24*24)/2=>10*(12*12)
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2)) # 維度: 10*(12*12)=>20*(8*8)/2=>20*4*4
        x = x.view(-1, 320) # 320=20*4*4
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)
