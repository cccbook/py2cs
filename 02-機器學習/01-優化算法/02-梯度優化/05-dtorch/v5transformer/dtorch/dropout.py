import torch
import torch.nn as nn

class Dropout(nn.Module):
    def __init__(self, p=0.5, inplace=False):
        super(Dropout, self).__init__()
        self.p = p
        self.inplace = inplace
        self.training = True  # 預設為訓練模式

    def forward(self, x, mask=None):
        """
        手動實現 Dropout。
        
        參數：
            x (torch.Tensor): 輸入張量。
        
        返回：
            torch.Tensor: Dropout 後的輸出張量。
        """
        if not self.training or self.p == 0:
            # 如果是測試模式或 p=0，直接返回輸入
            return x
        
        # 生成一個與輸入形狀相同的隨機 mask
        if mask is None:
            mask = (torch.rand_like(x) > self.p).float()
        
        if self.inplace:
            # 在原地修改輸入張量
            x.mul_(mask / (1 - self.p))
            return x
        else:
            # 返回新的張量
            return x * mask / (1 - self.p)
