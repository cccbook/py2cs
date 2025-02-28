import torch
import torch.nn as nn
import torch.nn.init as init

Module = nn.Module
CrossEntropyLoss = nn.CrossEntropyLoss

class Linear(Module):
    def __init__(self, in_features, out_features):
        super(Linear, self).__init__()
        # 初始化權重和偏置
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        # ====== 以下這段 kaiming_normal 非必要，但有的話，一開始 loss 就不會太高，效果會逼近 pytorch ===========
        init.kaiming_normal_(self.weight, mode='fan_in', nonlinearity='relu') # 使用 Kaiming 初始化權重
        init.zeros_(self.bias) # 初始化偏置為 0

    def forward(self, x):
        # 執行線性變換：y = xW^T + b
        return torch.matmul(x, self.weight.t()) + self.bias

class ReLU(Module):
    def __init__(self):
        super(ReLU, self).__init__()
    
    def forward(self, x):
        return x.clamp(min=0)
