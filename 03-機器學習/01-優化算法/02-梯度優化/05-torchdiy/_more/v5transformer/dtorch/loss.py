import torch
import torch.nn as nn

def logsumexp(x, dim=None, keepdim=False):
    """
    手動實現 logsumexp。
    
    參數：
        x (torch.Tensor): 輸入張量。
        dim (int): 沿哪個維度計算。如果為 None，則對整個張量計算。
        keepdim (bool): 是否保持維度。
    
    返回：
        torch.Tensor: 計算結果。
    """
    if dim is None:
        # 如果 dim 為 None，則對整個張量計算
        m = torch.max(x)
        return torch.log(torch.sum(torch.exp(x - m))) + m
    else:
        # 如果 dim 不為 None，則沿指定維度計算
        m, _ = torch.max(x, dim=dim, keepdim=True)
        result = torch.log(torch.sum(torch.exp(x - m), dim=dim, keepdim=keepdim)) + m
        return result.squeeze(dim) if not keepdim else result

class CrossEntropyLoss(nn.Module):
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
        log_softmax = logits - logsumexp(logits, dim=1, keepdim=True)

        # 2. 計算負對數似然損失（NLLLoss）
        # 使用 gather 來選取目標類別對應的 logit
        nll_loss = -log_softmax.gather(1, target.unsqueeze(1)).mean()
        
        return nll_loss