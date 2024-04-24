# https://ithelp.ithome.com.tw/articles/10218698
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

import sys

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
PATH = './model/cifar_net.pth'

# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def answer():
    # get some random training images
    dataiter = iter(trainloader)
    images, labels = next(dataiter)
    #images, labels = dataiter.next()

    # show images
    imshow(torchvision.utils.make_grid(images))
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

def train():
    net = Net()
    net.to(device) # 若有 GPU 會使用來加速

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # for epoch in range(2):  # loop over the dataset multiple times
    for epoch in range(3):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device) # GPU 加速版
            # inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')
    
    torch.save(net.state_dict(), PATH)

def predict():
    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    imshow(torchvision.utils.make_grid(images))
    print('Answer:', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

    net = Net()
    
    net.load_state_dict(torch.load(PATH))
    outputs = net(images)
    _, predicted = torch.max(outputs, 1)
    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))

def test():
    net = Net()
    net.load_state_dict(torch.load(PATH))

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

def init():
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # 注意： 根據 https://github.com/pytorch/pytorch/issues/28345 以下兩行 mac 版必須從 num_workers=2 改成 num_workers=0
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=0)
    return trainloader, testloader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Assuming that we are on a CUDA machine, this should print a CUDA device:
print('device=', device)

trainloader, testloader = init()
op = sys.argv[1]
if op == 'answer':
    answer()
elif op == 'train':
    train()
elif op == 'predict':
    predict()
elif op == 'test':
    test()
else:
    print('Error: op=', op)