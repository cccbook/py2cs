import torch
from transformer import Transformer

# 定義模型的超參數
input_size = 10
hidden_size = 20
num_heads = 2
num_layers = 2

# 建立模型
model = Transformer(input_size, hidden_size, num_heads, num_layers)

# 定義源序列和目標序列
src = torch.randn(1, 5, input_size)
tgt = torch.randn(1, 7, input_size)

# 在模型上進行推理
output = model(src, tgt)

# 輸出結果
print(output)
