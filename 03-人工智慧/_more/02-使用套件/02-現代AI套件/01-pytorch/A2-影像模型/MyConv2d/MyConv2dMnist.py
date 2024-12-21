import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# 定义自定义的卷积层 MyConv2d
class MyConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(MyConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # 初始化卷积核权重和偏置
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))
        
    def forward(self, x):
        batch_size, in_height, in_width = x.size(0), x.size(2), x.size(3)
        out_height = (in_height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (in_width + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        # 初始化输出
        output = torch.zeros(batch_size, self.out_channels, out_height, out_width)
        
        # 执行卷积操作
        for batch_idx in range(batch_size):
            for out_channel_idx in range(self.out_channels):
                for out_row in range(out_height):
                    for out_col in range(out_width):
                        # 计算输入和卷积核之间的元素积并累加
                        input_patch = x[batch_idx, :, out_row*self.stride:out_row*self.stride+self.kernel_size,
                                         out_col*self.stride:out_col*self.stride+self.kernel_size]
                        output[batch_idx, out_channel_idx, out_row, out_col] = torch.sum(
                            input_patch * self.weight[out_channel_idx]) + self.bias[out_channel_idx]
        
        return output

# 定义简单的卷积神经网络
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = MyConv2d(1, 32, kernel_size=5)
        self.conv2 = MyConv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(64 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 加载数据集
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# 初始化模型和优化器
net = ConvNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

# 训练模型
for epoch in range(5):  # 遍历数据集5次
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()

        # 正向传播
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        
        # 反向传播和优化
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 1 == 0:    # 每100批次打印一次损失
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

print('Finished Training')
