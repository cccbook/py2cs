import numpy as np

class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = np.array(data, dtype=np.float32)
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data) if requires_grad else None  # 初始化梯度
        self._backward = lambda: None
        self._prev = set()  # 用於存儲計算圖中的前一個節點

    def backward(self):
        if self.requires_grad:
            # 初始化輸出梯度為 1
            self.grad = np.ones_like(self.data)
            # 反向傳播，依次調用每個節點的 _backward
            self._backward()

    def __add__(self, other):
        if isinstance(other, Tensor):
            other = other.data
        out = Tensor(self.data + other, requires_grad=self.requires_grad)
        out._prev = {self}  # 記錄前一個節點
        print("add:forward, out = ", out.data)
        
        def _backward():
            if self.requires_grad:
                self.grad += out.grad
            print("add:backward, grad = ", self.grad)
            # 觸發前一個節點的反向傳播
            self._backward()
        out._backward = _backward
        return out

    def __mul__(self, other):
        if isinstance(other, Tensor):
            other = other.data
        out = Tensor(self.data * other, requires_grad=self.requires_grad)
        out._prev = {self}  # 記錄前一個節點
        
        def _backward():
            if self.requires_grad:
                self.grad += out.grad * other
            print("mul:backward, grad = ", self.grad)
            # 觸發前一個節點的反向傳播
            self._backward()
        out._backward = _backward
        return out

    def __matmul__(self, other):
        if isinstance(other, Tensor):
            other = other.data
        out = Tensor(self.data @ other, requires_grad=self.requires_grad)
        out._prev = {self}  # 記錄前一個節點
        print("matmul:forward, out = ", out.data)
        
        def _backward():
            if self.requires_grad:
                self.grad += out.grad @ other.T
            print("matmul:backward, grad = ", self.grad)
            # 觸發前一個節點的反向傳播
            self._backward()
        out._backward = _backward
        return out

    def relu(self):
        out = Tensor(np.maximum(0, self.data), requires_grad=self.requires_grad)
        out._prev = {self}  # 記錄前一個節點
        
        def _backward():
            if self.requires_grad:
                self.grad += out.grad * (out.data > 0)
            print("relu:backward, grad = ", self.grad)
            # 觸發前一個節點的反向傳播
            self._backward()
        out._backward = _backward
        return out

class Linear:
    def __init__(self, in_features, out_features):
        self.weights = Tensor(np.random.randn(in_features, out_features), requires_grad=True)
        self.bias = Tensor(np.random.randn(out_features), requires_grad=True)

    def __call__(self, x):
        return x @ self.weights + self.bias