from macrograd.engine import Tensor
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

# 使用 torchvision 下載 MNIST 資料集
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# 將資料轉換為 NumPy 陣列
train_images = train_dataset.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
train_labels = train_dataset.targets.numpy()
test_images = test_dataset.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
test_labels = test_dataset.targets.numpy()

# 將標籤轉換為 one-hot 編碼
def to_categorical(y, num_classes):
    return np.eye(num_classes)[y]

y_train = to_categorical(train_labels, 10)

def forward(X, Y, W):
    y_predW = X.matmul(W)
    probs = y_predW.softmax()
    loss = probs.cross_entropy(Y)
    return loss

batch_size = 32
steps = 20000

X = Tensor(train_images); Y = Tensor(y_train) # 全部資料
# new initialized weights for gradient descent
Wb = Tensor(np.random.randn(784, 10))
for step in range(steps):
    ri = np.random.permutation(train_images.shape[0])[:batch_size]
    Xb, yb = Tensor(train_images[ri]), Tensor(y_train[ri]) # Batch 資料
    lossb = forward(Xb, yb, Wb)
    lossb.backward()
    if step % 1000 == 0 or step == steps-1:
        loss = forward(X, Y, Wb).data/X.data.shape[0]
        print(f'loss in step {step} is {loss}')
    Wb.data = Wb.data - 0.01*Wb.grad # update weights, 相當於 optimizer.step()
    Wb.grad = 0

from sklearn.metrics import accuracy_score
print(f'accuracy on test data is {accuracy_score(np.argmax(np.matmul(test_images,Wb.data),axis = 1),test_labels)*100} %')