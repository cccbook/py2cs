from dsnn import *
import torch
from torchvision import datasets, transforms

# 下載 MNIST 數據集
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# 轉換為 numpy 數組
train_data = train_dataset.data.numpy().astype(np.float32) / 255.0
train_labels = train_dataset.targets.numpy()
test_data = test_dataset.data.numpy().astype(np.float32) / 255.0
test_labels = test_dataset.targets.numpy()

class Model:
    def __init__(self):
        # 定義模型的層
        self.conv1 = Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = MaxPool2d(kernel_size=2, stride=2)
        self.flatten = Flatten()
        self.fc1 = Linear(32 * 14 * 14, 10)

    def __call__(self, x):
        # 前向傳播
        x = self.conv1(x)
        x = x.relu()
        x = self.pool1(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return x

    def parameters(self):
        """
        返回模型中所有需要訓練的參數。
        """
        params = []
        # 添加卷積層的參數
        params.append(self.conv1.weights)
        params.append(self.conv1.bias)
        # 添加全連接層的參數
        params.append(self.fc1.weights)
        params.append(self.fc1.bias)
        return params

# 訓練模型
model = Model()
criterion = CrossEntropyLoss()
optimizer = SGD(model.parameters(), lr=0.001, max_norm=1.0)  # 降低學習率，添加梯度裁剪

for epoch in range(5):
    for i in range(len(train_data)):
        x = Tensor(train_data[i:i+1].reshape(1, 1, 28, 28), requires_grad=True)
        y = Tensor(train_labels[i:i+1], requires_grad=False)

        # 前向傳播
        output = model(x)
        loss = criterion(output, y)

        # 反向傳播
        loss.backward()

        # 更新參數
        optimizer.step()

        if i % 100 == 0:
            print(f"Epoch {epoch}, Step {i}, Loss: {loss.data}")

# 測試模型
correct = 0
total = 0
for i in range(len(test_data)):
    x = Tensor(test_data[i:i+1].reshape(1, 1, 28, 28), requires_grad=False)
    y = test_labels[i:i+1]

    output = model(x)
    pred = np.argmax(output.data)
    if pred == y:
        correct += 1
    total += 1

print(f"Test Accuracy: {correct / total * 100:.2f}%")
