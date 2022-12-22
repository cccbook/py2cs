import numpy as np
import torch 
import torch.nn as nn
import torchvision

model = torch.load('results/model.ckpt')

batch_size_test = 1000
test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('files/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
    batch_size=batch_size_test, shuffle=True)

examples = enumerate(test_loader)

batch_idx, (example_data, example_targets) = next(examples)

import matplotlib.pyplot as plt

fig = plt.figure()
for i in range(6):
    data = example_data[i][0]
    plt.subplot(2,3,i+1)
    plt.tight_layout()
    output = model(data)
    plt.imshow(data, cmap='gray', interpolation='none')
    plt.title("Ground Truth: {}".format(example_targets[i]))
    plt.xticks([])
    plt.yticks([])

plt.show()