# 來源 -- https://kknews.cc/code/zp8megl.html
# 導入必要的庫 
import torch
from pytorch_transformers import GPT2Tokenizer, GPT2LMHeadModel
# 加載預訓練模型tokenizer (vocabulary)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# 對文本輸入進行編碼
text = "What is the fastest car in the"
indexed_tokens = tokenizer.encode(text)
# 在PyTorch張量中轉換indexed_tokens
tokens_tensor = torch.tensor([indexed_tokens]) 
# 加載預訓練模型 (weights) 
model = GPT2LMHeadModel.from_pretrained('gpt2') 
#將模型設置為evaluation模式，關閉DropOut模塊 
model.eval() # 如果你有GPU，把所有東西都放在cuda上 
# tokens_tensor = tokens_tensor.to('cuda')
# model.to('cuda') 
# 預測所有的tokens 

with torch.no_grad():
    outputs = model(tokens_tensor)
    predictions = outputs[0]
# 得到預測的單詞 
predicted_index = torch.argmax(predictions[0, -1, :]).item()
predicted_text = tokenizer.decode(indexed_tokens + [predicted_index])

# 列印預測單詞
print(predicted_text)
