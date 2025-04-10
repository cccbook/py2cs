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
