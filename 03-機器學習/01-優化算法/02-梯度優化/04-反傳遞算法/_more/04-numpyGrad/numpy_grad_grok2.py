import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 基本張量類（保持不變）
class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = np.array(data)
        self.requires_grad = requires_grad
        self.grad = None
        self._backward = lambda: None
        
    def backward(self, grad=None):
        if not self.requires_grad:
            return
        if grad is None:
            grad = np.ones_like(self.data)
        if self.grad is None:
            self.grad = grad
        else:
            self.grad += grad
        self._backward()

# Linear 層（保持不變）
class Linear:
    def __init__(self, in_features, out_features):
        self.weight = Tensor(np.random.randn(in_features, out_features) * 0.01, 
                          requires_grad=True)
        self.bias = Tensor(np.zeros(out_features), 
                         requires_grad=True)
        
    def forward(self, x):
        if not isinstance(x, Tensor):
            x = Tensor(x, requires_grad=True)
        out = Tensor(np.dot(x.data, self.weight.data) + self.bias.data,
                    requires_grad=True)
        def backward():
            if out.grad is None:
                return
            x_grad = np.dot(out.grad, self.weight.data.T)
            w_grad = np.dot(x.data.T, out.grad)
            b_grad = np.sum(out.grad, axis=0)
            x.backward(x_grad)
            self.weight.backward(w_grad)
            self.bias.backward(b_grad)
        out._backward = backward
        return out

# ReLU 層（保持不變）
class ReLU:
    def forward(self, x):
        if not isinstance(x, Tensor):
            x = Tensor(x, requires_grad=True)
        out = Tensor(np.maximum(0, x.data), requires_grad=True)
        def backward():
            if out.grad is None:
                return
            x_grad = out.grad * (x.data > 0)
            x.backward(x_grad)
        out._backward = backward
        return out

# Softmax 層
class Softmax:
    def forward(self, x):
        if not isinstance(x, Tensor):
            x = Tensor(x, requires_grad=True)
        # 防止數值溢出
        shifted = x.data - np.max(x.data, axis=1, keepdims=True)
        exp_x = np.exp(shifted)
        out = Tensor(exp_x / np.sum(exp_x, axis=1, keepdims=True),
                    requires_grad=True)
        
        def backward():
            if out.grad is None:
                return
            # Softmax 梯度計算
            softmax = out.data
            grad = np.zeros_like(softmax)
            for i in range(len(softmax)):
                sm = softmax[i]
                grad[i] = sm * (out.grad[i] - np.sum(sm * out.grad[i]))
            x.backward(grad)
        out._backward = backward
        return out

# 交叉熵損失
def cross_entropy_loss(preds, targets):
    if not isinstance(preds, Tensor):
        preds = Tensor(preds, requires_grad=True)
    targets = np.array(targets)
    
    # 計算損失
    n = preds.data.shape[0]
    log_probs = -np.log(preds.data[range(n), targets] + 1e-10)
    loss = Tensor(np.mean(log_probs), requires_grad=True)
    
    def backward():
        if loss.grad is None:
            return
        grad = np.zeros_like(preds.data)
        grad[range(n), targets] = -1.0 / (preds.data[range(n), targets] + 1e-10) / n
        preds.backward(grad * loss.grad)
    loss._backward = backward
    return loss

# 簡單的神經網絡
class SimpleNet:
    def __init__(self):
        self.layers = [
            Linear(784, 128),
            ReLU(),
            Linear(128, 64),
            ReLU(),
            Linear(64, 10),
            Softmax()
        ]
    
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def zero_grad(self):
        for layer in self.layers:
            if hasattr(layer, 'weight'):
                layer.weight.grad = None
                layer.bias.grad = None

# 獲取 MNIST 數據
def get_mnist():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    return train_loader, test_loader

# 訓練函數
def train(model, train_loader, epochs=3, lr=0.001):
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            # 準備數據
            data = data.numpy().reshape(-1, 784)  # 展平為 (batch_size, 784)
            target = target.numpy()
            
            # 前向傳播
            model.zero_grad()
            output = model.forward(data)
            loss = cross_entropy_loss(output, target)
            
            # 反向傳播
            loss.backward()
            
            # 更新參數
            for layer in model.layers:
                if hasattr(layer, 'weight'):
                    layer.weight.data -= lr * layer.weight.grad
                    layer.bias.data -= lr * layer.bias.grad
            
            # 統計
            total_loss += loss.data
            pred = np.argmax(output.data, axis=1)
            correct += np.sum(pred == target)
            total += len(target)
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.data:.4f}')
        
        print(f'Epoch {epoch} finished, Avg Loss: {total_loss/len(train_loader):.4f}, '
              f'Accuracy: {100. * correct/total:.2f}%')

# 主程序
if __name__ == "__main__":
    # 初始化網絡和數據
    model = SimpleNet()
    train_loader, test_loader = get_mnist()
    
    # 訓練模型
    train(model, train_loader)
