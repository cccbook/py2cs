import torch
import torch.nn as nn
import torch.nn.functional as F

# Conv2d = nn.Conv2d
# MaxPool2d = nn.MaxPool2d

class MaxPool2d(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super(MaxPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def forward(self, x):
        """
        手動實現 2D 最大池化。
        
        參數：
            x (torch.Tensor): 輸入張量，形狀為 (batch_size, channels, height, width)。
        
        返回：
            torch.Tensor: 池化後的輸出張量。
        """
        # 獲取輸入的形狀
        batch_size, channels, height, width = x.shape
        
        # 計算輸出形狀
        out_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        # 初始化輸出張量
        output = torch.zeros((batch_size, channels, out_height, out_width), device=x.device)
        
        # 對輸入進行填充
        if self.padding > 0:
            x = F.pad(x, (self.padding, self.padding, self.padding, self.padding))
        
        # 遍歷每個池化窗口
        for i in range(out_height):
            for j in range(out_width):
                # 計算窗口的起始和結束位置
                h_start = i * self.stride
                h_end = h_start + self.kernel_size
                w_start = j * self.stride
                w_end = w_start + self.kernel_size
                
                # 提取窗口並計算最大值
                window = x[:, :, h_start:h_end, w_start:w_end]
                output[:, :, i, j] = window.max(dim=2).values.max(dim=2).values
        
        return output

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(Conv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.use_bias = bias
        
        # 初始化卷積核權重
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, self.kernel_size[0], self.kernel_size[1]))
        # 初始化偏置
        if self.use_bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        """
        手動實現 2D 卷積。
        
        參數：
            x (torch.Tensor): 輸入張量，形狀為 (batch_size, in_channels, height, width)。
        
        返回：
            torch.Tensor: 卷積後的輸出張量。
        """
        # 獲取輸入的形狀
        batch_size, in_channels, height, width = x.shape
        
        # 計算輸出形狀
        out_height = (height + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[0] + 1
        out_width = (width + 2 * self.padding[1] - self.kernel_size[1]) // self.stride[1] + 1
        
        # 初始化輸出張量
        output = torch.zeros((batch_size, self.out_channels, out_height, out_width), device=x.device)
        
        # 對輸入進行填充
        if self.padding[0] > 0 or self.padding[1] > 0:
            x = F.pad(x, (self.padding[1], self.padding[1], self.padding[0], self.padding[0]))
        
        # 遍歷每個輸出位置
        for i in range(out_height):
            for j in range(out_width):
                # 計算窗口的起始和結束位置
                h_start = i * self.stride[0]
                h_end = h_start + self.kernel_size[0]
                w_start = j * self.stride[1]
                w_end = w_start + self.kernel_size[1]
                
                # 提取窗口
                window = x[:, :, h_start:h_end, w_start:w_end]
                
                # 對每個輸出通道進行卷積
                for k in range(self.out_channels):
                    output[:, k, i, j] = torch.sum(window * self.weight[k], dim=(1, 2, 3))
                    if self.use_bias:
                        output[:, k, i, j] += self.bias[k]
        
        return output

