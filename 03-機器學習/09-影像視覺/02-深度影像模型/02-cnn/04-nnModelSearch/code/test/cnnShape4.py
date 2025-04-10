import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

def get_output_shape(model, image_dim):
    return model(torch.rand(*(image_dim))).data.shape

def test():
    expected_input_shape = (1, 1, 192, 168)
    x = torch.randn(expected_input_shape)
    print("x:", x.shape)
    conv1 = nn.Conv2d(1, 32, 3, 1)
    x = conv1(x)
    print("conv2d(1,32,3,1):", x.shape)
    conv2 = nn.Conv2d(32, 64, 3, 1)
    x = conv2(x)
    print("conv2d(32,64,3,1):", x.shape)
    dropout1 = nn.Dropout2d(0.25)
    x = dropout1(x)
    print("Dropout2d(0.25):", x.shape)
    dropout2 = nn.Dropout2d(0.5)
    x = dropout2(x)
    print("dropout2d(0.5):", x.shape)
    maxpool1 = nn.MaxPool2d(2)
    x = maxpool1(x)
    print("MaxPool2d(2):", x.shape)
    maxpool2 = nn.MaxPool2d(3)
    x = maxpool2(x)
    print("MaxPool2d(3):", x.shape)

test()
