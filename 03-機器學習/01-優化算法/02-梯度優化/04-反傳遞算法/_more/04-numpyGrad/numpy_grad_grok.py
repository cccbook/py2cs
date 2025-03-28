import numpy as np

# 基本張量類，帶自動梯度計算
class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = np.array(data)
        self.requires_grad = requires_grad
        self.grad = None
        self._backward = lambda: None  # 儲存反向傳播函數
        
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

# Linear 層
class Linear:
    def __init__(self, in_features, out_features):
        # 初始化權重和偏差
        self.weight = Tensor(np.random.randn(in_features, out_features) * 0.01, 
                          requires_grad=True)
        self.bias = Tensor(np.zeros(out_features), 
                         requires_grad=True)
        
    def forward(self, x):
        if not isinstance(x, Tensor):
            x = Tensor(x, requires_grad=True)
            
        # 向前傳播: y = xW + b
        out = Tensor(np.dot(x.data, self.weight.data) + self.bias.data,
                    requires_grad=True)
        
        # 定義反向傳播
        def backward():
            if out.grad is None:
                return
                
            # 計算輸入的梯度 dx = dy * W^T
            x_grad = np.dot(out.grad, self.weight.data.T)
            # 計算權重的梯度 dW = x^T * dy
            w_grad = np.dot(x.data.T, out.grad)
            # 計算偏差的梯度 db = sum(dy)
            b_grad = np.sum(out.grad, axis=0)
            
            # 傳遞梯度
            x.backward(x_grad)
            self.weight.backward(w_grad)
            self.bias.backward(b_grad)
            
        out._backward = backward
        return out

# ReLU 層
class ReLU:
    def forward(self, x):
        if not isinstance(x, Tensor):
            x = Tensor(x, requires_grad=True)
            
        # 向前傳播: max(0, x)
        out = Tensor(np.maximum(0, x.data),
                    requires_grad=True)
        
        # 定義反向傳播
        def backward():
            if out.grad is None:
                return
            # ReLU 的梯度: dy * (x > 0)
            x_grad = out.grad * (x.data > 0)
            x.backward(x_grad)
            
        out._backward = backward
        return out

# 使用範例
def example():
    # 創建樣本數據
    x = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), requires_grad=True)
    
    # 創建層
    linear = Linear(2, 3)
    relu = ReLU()
    
    # 向前傳播
    y = linear.forward(x)
    z = relu.forward(y)
    
    # 假設損失函數對 z 的梯度
    z.backward(np.ones_like(z.data))
    
    # 打印結果
    print("Input:\n", x.data)
    print("Linear output:\n", y.data)
    print("ReLU output:\n", z.data)
    print("Gradient of input:\n", x.grad)
    print("Gradient of weights:\n", linear.weight.grad)
    print("Gradient of bias:\n", linear.bias.grad)

if __name__ == "__main__":
    example()