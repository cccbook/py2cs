import numpy as np

class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = np.array(data, dtype=np.float32)
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data) if requires_grad else None  # 初始化梯度
        self._backward = lambda: None

    def backward(self):
        if self.requires_grad:
            self.grad = np.ones_like(self.data)  # 初始化輸出梯度為 1
            self._backward()

    def __add__(self, other):
        if isinstance(other, Tensor):
            other = other.data
        out = Tensor(self.data + other, requires_grad=self.requires_grad)
        
        def _backward():
            if self.requires_grad:
                self.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        if isinstance(other, Tensor):
            other = other.data
        out = Tensor(self.data * other, requires_grad=self.requires_grad)
        
        def _backward():
            if self.requires_grad:
                self.grad += out.grad * other
        out._backward = _backward
        return out

    def __matmul__(self, other):
        if isinstance(other, Tensor):
            other = other.data
        out = Tensor(self.data @ other, requires_grad=self.requires_grad)
        
        def _backward():
            if self.requires_grad:
                self.grad += out.grad @ other.T
        out._backward = _backward
        return out

    def relu(self):
        out = Tensor(np.maximum(0, self.data), requires_grad=self.requires_grad)
        
        def _backward():
            if self.requires_grad:
                self.grad += out.grad * (out.data > 0)
        out._backward = _backward
        return out

class Linear:
    def __init__(self, in_features, out_features):
        self.weights = Tensor(np.random.randn(in_features, out_features), requires_grad=True)
        self.bias = Tensor(np.random.randn(out_features), requires_grad=True)

    def __call__(self, x):
        return x @ self.weights + self.bias