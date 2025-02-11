import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# 設定 MPS 為設備
device = torch.device("mps") if torch.backends.mps.is_built() else torch.device("cpu")

# 載入 GPT-2 模型和 tokenizer
model_name = "gpt2"  # 可以換成其他 GPT-2 變體，例如 "gpt2-medium"
model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# 定義輸入 prompt
# prompt = "今天的天氣真好，我們來談談人工智慧的未來。"
prompt = "once upon a time"

# 將 prompt 編碼成 token 並移動到 MPS 設備上
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

# 使用 GPT-2 模型生成文本
output = model.generate(input_ids, max_length=100, temperature=0.7, top_k=50)

# 解碼生成的 token 並顯示結果
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
