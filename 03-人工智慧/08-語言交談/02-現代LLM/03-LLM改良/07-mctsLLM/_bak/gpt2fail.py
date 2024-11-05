import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from torch.nn import Embedding
import numpy as np

# 定義 GPT-2 簡單模型
class SimpleGPT2(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers):
        super(SimpleGPT2, self).__init__()
        self.embedding = Embedding(vocab_size, d_model)
        decoder_layers = TransformerDecoderLayer(d_model, nhead)
        self.transformer_decoder = TransformerDecoder(decoder_layers, num_layers)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x) * np.sqrt(self.embedding.embedding_dim)  # Embedding scaling
        x = self.transformer_decoder(x, x)
        return self.fc(x)

# 設定參數
vocab_size = 50257  # GPT-2 的詞彙表大小
d_model = 768  # 嵌入維度
nhead = 12  # 注意力頭數
num_layers = 12  # 編碼層數
max_length = 100  # 生成文本的最大長度

# 初始化模型
model = SimpleGPT2(vocab_size, d_model, nhead, num_layers).to('mps')  # 使用 MPS 設備

# 定義生成文本的函數
def generate_text(model, prompt, max_length):
    model.eval()  # 切換到評估模式
    input_ids = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to('mps')  # 編碼 prompt

    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(input_ids)  # 前向推理
            next_token_logits = outputs[:, -1, :]  # 獲取最後一個 token 的 logits
            next_token = torch.multinomial(F.softmax(next_token_logits, dim=-1), num_samples=1)  # 按概率抽樣
            input_ids = torch.cat((input_ids, next_token.unsqueeze(0)), dim=1)  # 更新輸入序列

    return input_ids

# 使用示例
prompt = "今天的天氣真好，我們來談談人工智慧的未來。"
generated_ids = generate_text(model, prompt, max_length)
generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print(generated_text)
