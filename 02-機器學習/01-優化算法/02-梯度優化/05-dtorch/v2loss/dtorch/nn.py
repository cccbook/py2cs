import torch
import torch.nn as nn
import torch.nn.init as init
from . import loss

Module = nn.Module
# CrossEntropyLoss = nn.CrossEntropyLoss

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

class CrossEntropyLoss(Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, logits, target):
        """
        計算 CrossEntropyLoss。
        
        參數：
            logits (torch.Tensor): 模型的輸出，形狀為 (batch_size, num_classes)。
            target (torch.Tensor): 目標類別，形狀為 (batch_size,)。
        
        返回：
            loss (torch.Tensor): 計算得到的損失值。
        """
        # 1. 計算 LogSoftmax
        # log_softmax = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        log_softmax = logits - loss.logsumexp(logits, dim=1, keepdim=True)

        # 2. 計算負對數似然損失（NLLLoss）
        # 使用 gather 來選取目標類別對應的 logit
        nll_loss = -log_softmax.gather(1, target.unsqueeze(1)).mean()
        
        return nll_loss