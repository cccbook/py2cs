import numpy as np

class Tensor:
    def __init__(self, data, _children=(), _op=''):
        self.data = np.array(data)
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"

    def backward(self):
        # 拓撲排序所有子節點
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        
        # 從輸出節點開始反向傳播
        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            node._backward()
    
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), '+')
        
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        
        return out
    
    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data - other.data, (self, other), '-')
        
        def _backward():
            self.grad += out.grad
            other.grad -= out.grad  # 減法運算對第二個操作數梯度取負
        out._backward = _backward
        
        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), '*')
        
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        
        return out
    
    def __truediv__(self, other):  # 新增除法操作
        if isinstance(other, (int, float)):  # 支持除以純數值
            out = Tensor(self.data / other, (self,), '/')
            
            def _backward():
                self.grad += out.grad / other
            out._backward = _backward
            
            return out
        
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data / other.data, (self, other), '/')
        
        def _backward():
            self.grad += out.grad / other.data
            other.grad += -self.data * out.grad / (other.data * other.data)
        out._backward = _backward
        
        return out
    
    def __matmul__(self, other):  # 矩陣乘法
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data, (self, other), '@')
        
        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad
        out._backward = _backward
        
        return out
    
    def relu(self):
        out = Tensor(np.maximum(0, self.data), (self,), 'ReLU')
        
        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        
        return out
    
    def pow(self, power):  # 新增冪次方操作
        out = Tensor(self.data ** power, (self,), f'pow{power}')
        
        def _backward():
            self.grad += (power * self.data ** (power - 1)) * out.grad
        out._backward = _backward
        
        return out


class Linear:
    def __init__(self, in_features, out_features):
        # 使用 He 初始化
        self.weight = Tensor(
            np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)
        )
        self.bias = Tensor(np.zeros(out_features))
        self.parameters = [self.weight, self.bias]
    
    def __call__(self, x):
        return x @ self.weight + self.bias


class ReLU:
    def __call__(self, x):
        return x.relu()


class SGD:
    def __init__(self, parameters, lr=0.01):
        self.parameters = parameters
        self.lr = lr
    
    def step(self):
        for p in self.parameters:
            p.data -= self.lr * p.grad
    
    def zero_grad(self):
        for p in self.parameters:
            p.grad = np.zeros_like(p.data)


# 計算 MSE 損失
def mse_loss(pred, target):
    # 使用簡單的平方誤差計算，避免使用 mean() 和 sum()
    diff = pred - target
    squared = diff * diff  # 逐元素平方
    
    # 創建一個新的張量作為損失
    loss = Tensor(np.mean(squared.data), (squared,), 'mse_loss')
    
    def _backward():
        # 手動計算平方誤差對原張量的梯度
        n = np.prod(pred.data.shape)  # 元素總數
        diff.grad += 2 * diff.data * loss.grad / n
    
    loss._backward = _backward
    return loss


# 範例使用
def example():
    # 創建一個簡單的網絡：Linear -> ReLU -> Linear
    linear1 = Linear(3, 4)
    relu = ReLU()
    linear2 = Linear(4, 1)
    
    # 獲取所有參數
    parameters = linear1.parameters + linear2.parameters
    
    # 創建優化器
    optimizer = SGD(parameters, lr=0.01)
    
    # 輸入數據
    x = Tensor(np.random.randn(5, 3))  # 批次大小為5，特徵數為3
    y = Tensor(np.random.randn(5, 1))  # 目標
    
    # 訓練循環
    for i in range(10):
        # 前向傳播
        h = linear1(x)
        h = relu(h)
        pred = linear2(h)
        
        # 計算 MSE 損失
        loss = mse_loss(pred, y)
        print(f"Epoch {i}, Loss: {loss.data}")
        
        # 反向傳播
        optimizer.zero_grad()
        loss.backward()
        
        # 更新參數
        optimizer.step()
    
    return loss


# 另一種 MSE 損失計算方式（使用平方冪次方）
def mse_loss_alternative(pred, target):
    return (pred - target).pow(2).data.mean()


# 簡單測試 - 確保基本操作正常工作
def test_basics():
    # 測試基本操作
    a = Tensor([1, 2, 3])
    b = Tensor([4, 5, 6])
    c = a + b
    print(f"a + b = {c.data}")
    
    d = a * b
    print(f"a * b = {d.data}")
    
    e = a - b
    print(f"a - b = {e.data}")
    
    # 測試梯度
    f = (a * b).pow(2)
    f.backward()
    print(f"Gradient of a: {a.grad}")
    print(f"Gradient of b: {b.grad}")


if __name__ == "__main__":
    print("Running basic tests...")
    test_basics()
    print("\nRunning training example...")
    example()