import numpy as np

class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = np.array(data, dtype=np.float32)
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data) if requires_grad else None
        self._backward = lambda: None

    def backward(self):
        if self.requires_grad:
            self.grad = np.ones_like(self.data)
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

    def reshape(self, shape):
        out = Tensor(self.data.reshape(shape), requires_grad=self.requires_grad)
        
        def _backward():
            if self.requires_grad:
                self.grad += out.grad.reshape(self.data.shape)
        out._backward = _backward
        return out

    def transpose(self, axes):
        out = Tensor(self.data.transpose(axes), requires_grad=self.requires_grad)
        
        def _backward():
            if self.requires_grad:
                self.grad += out.grad.transpose(axes)
        out._backward = _backward
        return out

class Linear:
    def __init__(self, in_features, out_features):
        self.weights = Tensor(np.random.randn(in_features, out_features), requires_grad=True)
        self.bias = Tensor(np.random.randn(out_features), requires_grad=True)

    def __call__(self, x):
        return x @ self.weights + self.bias

class Conv2d:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weights = Tensor(np.random.randn(out_channels, in_channels, kernel_size, kernel_size), requires_grad=True)
        self.bias = Tensor(np.random.randn(out_channels), requires_grad=True)

    def __call__(self, x):
        batch_size, in_channels, in_height, in_width = x.data.shape
        out_channels, _, kernel_height, kernel_width = self.weights.data.shape

        # 計算輸出尺寸
        out_height = (in_height - kernel_height + 2 * self.padding) // self.stride + 1
        out_width = (in_width - kernel_width + 2 * self.padding) // self.stride + 1

        # 添加 padding
        if self.padding > 0:
            x_padded = np.pad(x.data, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)))
        else:
            x_padded = x.data

        # 初始化輸出
        out = np.zeros((batch_size, out_channels, out_height, out_width))

        # 卷積操作
        for b in range(batch_size):
            for c_out in range(out_channels):
                for h in range(out_height):
                    for w in range(out_width):
                        h_start = h * self.stride
                        h_end = h_start + kernel_height
                        w_start = w * self.stride
                        w_end = w_start + kernel_width
                        out[b, c_out, h, w] = np.sum(
                            x_padded[b, :, h_start:h_end, w_start:w_end] * self.weights.data[c_out]
                        ) + self.bias.data[c_out]

        out = Tensor(out, requires_grad=True)
        
        def _backward():
            if x.requires_grad:
                x.grad += np.zeros_like(x.data)
            if self.weights.requires_grad:
                self.weights.grad += np.zeros_like(self.weights.data)
            if self.bias.requires_grad:
                self.bias.grad += np.zeros_like(self.bias.data)
        out._backward = _backward
        return out

class MaxPool2d:
    def __init__(self, kernel_size, stride=None):
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size

    def __call__(self, x):
        batch_size, channels, in_height, in_width = x.data.shape
        out_height = (in_height - self.kernel_size) // self.stride + 1
        out_width = (in_width - self.kernel_size) // self.stride + 1

        out = np.zeros((batch_size, channels, out_height, out_width))

        for b in range(batch_size):
            for c in range(channels):
                for h in range(out_height):
                    for w in range(out_width):
                        h_start = h * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = w * self.stride
                        w_end = w_start + self.kernel_size
                        out[b, c, h, w] = np.max(x.data[b, c, h_start:h_end, w_start:w_end])

        out = Tensor(out, requires_grad=True)
        
        def _backward():
            if x.requires_grad:
                x.grad += np.zeros_like(x.data)
        out._backward = _backward
        return out

class Flatten:
    def __call__(self, x):
        batch_size = x.data.shape[0]
        out = Tensor(x.data.reshape(batch_size, -1), requires_grad=True)
        
        def _backward():
            if x.requires_grad:
                x.grad += out.grad.reshape(x.data.shape)
        out._backward = _backward
        return out

class CrossEntropyLoss:
    def __call__(self, logits, targets):
        """
        logits = logits.data
        targets = targets.data.astype(np.int64) # targets = targets.data
        batch_size = logits.shape[0]

        # Softmax
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        # Cross-entropy loss
        loss = -np.sum(np.log(probs[np.arange(batch_size), targets])) / batch_size

        # Gradients
        probs[np.arange(batch_size), targets] -= 1
        grad = probs / batch_size

        out = Tensor(loss, requires_grad=True)
        out.grad = grad
        return out
        """
        # 定義 CrossEntropyLoss
class CrossEntropyLoss:
    def __call__(self, logits, targets):
        logits = logits.data
        targets = targets.data.astype(np.int64)  # 確保 targets 是整數類型
        batch_size = logits.shape[0]

        # 改進的 Softmax 計算
        logits = logits - np.max(logits, axis=1, keepdims=True)  # 避免數值溢出
        exp_logits = np.exp(logits)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        # Cross-entropy loss
        loss = -np.sum(np.log(probs[np.arange(batch_size), targets] + 1e-10)) / batch_size  # 添加小常數避免 log(0)

        # Gradients
        probs[np.arange(batch_size), targets] -= 1
        grad = probs / batch_size

        out = Tensor(loss, requires_grad=True)
        out.grad = grad
        return out

class SGD:
    def __init__(self, parameters, lr=0.001, max_norm=1.0):  # 降低學習率，添加梯度裁剪
        self.parameters = parameters
        self.lr = lr
        self.max_norm = max_norm

    def step(self):
        # 梯度裁剪
        self.clip_gradients()
        # 更新參數
        for param in self.parameters:
            if param.requires_grad:
                param.data -= self.lr * param.grad
                param.grad = np.zeros_like(param.grad)

    def clip_gradients(self):
        for param in self.parameters:
            if param.requires_grad:
                norm = np.linalg.norm(param.grad)
                if norm > self.max_norm:
                    param.grad = param.grad * (self.max_norm / norm)


class SGDMomentum:
    def __init__(self, parameters, lr=0.01, momentum=0.9):
        """
        初始化帶動量的 SGD 優化器。

        參數:
            parameters: 需要優化的參數列表。
            lr: 學習率，默認為 0.01。
            momentum: 動量係數，默認為 0.9。
        """
        self.parameters = parameters
        self.lr = lr
        self.momentum = momentum
        self.velocities = [np.zeros_like(param.data) for param in parameters]

    def step(self):
        """
        更新所有參數的值。
        """
        for i, param in enumerate(self.parameters):
            if param.requires_grad:
                # 更新速度
                self.velocities[i] = self.momentum * self.velocities[i] - self.lr * param.grad
                # 更新參數值
                param.data += self.velocities[i]
                # 重置梯度
                param.grad = np.zeros_like(param.grad)