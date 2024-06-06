import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

def get_output_shape(model, image_dim):
    return model(torch.rand(*(image_dim))).data.shape

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.expected_input_shape = (1, 1, 192, 168)
        x = torch.randn(self.expected_input_shape)
        print("x:", x.shape)
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        x = self.conv1(x)
        print("conv2d(1,32,3,1):", x.shape)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        x = self.conv2(x)
        print("conv2d(32,64,3,1):", x.shape)
        self.dropout1 = nn.Dropout2d(0.25)
        x = self.dropout1(x)
        print("Dropout2d(0.25):", x.shape)
        self.dropout2 = nn.Dropout2d(0.5)
        x = self.dropout2(x)
        print("dropout2d(0.5):", x.shape)
        self.maxpool1 = nn.MaxPool2d(2)
        x = self.maxpool1(x)
        print("MaxPool2d(2):", x.shape)
        self.maxpool2 = nn.MaxPool2d(3)
        x = self.maxpool2(x)
        print("MaxPool2d(3):", x.shape)

        # Calculate the input of the Linear layer
        conv1_out = get_output_shape(self.maxpool1, get_output_shape(self.conv1, self.expected_input_shape))
        conv2_out = get_output_shape(self.maxpool2, get_output_shape(self.conv2, conv1_out)) 
        fc1_in = np.prod(list(conv2_out)) # Flatten

        self.fc1 = nn.Linear(fc1_in, 38)

    def forward(self, x):
        print("x:", x.shape)
        x = self.conv1(x)
        print("conv1(x):", x.shape)
        x = F.relu(x)
        print("relu(x):", x.shape)
        x = self.maxpool1(x) 
        print("max_poll(x):", x.shape)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.maxpool2(x) 
        x = self.dropout1(x) 
        x = torch.flatten(x, 1) # flatten to a single dimension
        x = self.fc1(x) 
        output = F.log_softmax(x, dim=1) 
        return output

net = Net()
x = torch.randn(1, 1, 192, 168)
# net.forward(x)
