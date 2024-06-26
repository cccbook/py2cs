import torch
import torch.nn as nn

# 定義一個簡單的ReLU層
class ReLULayer(nn.Module):
    def __init__(self):
        super(ReLULayer, self).__init__()

    def forward(self, x):
        return torch.max(torch.zeros_like(x), x)

# 測試ReLU層
if __name__ == "__main__":
    # 建立一個輸入張量
    x = torch.tensor([-1.0, 0.0, 1.0, 2.0])

    # 建立ReLU層
    relu_layer = ReLULayer()

    # 將輸入張量通過ReLU層
    output = relu_layer(x)

    print("輸入張量:", x)
    print("ReLU後的輸出:", output)
