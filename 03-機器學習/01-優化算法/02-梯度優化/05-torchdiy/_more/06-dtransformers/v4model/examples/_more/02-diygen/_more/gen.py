from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

# 加載 DistilGPT-2 模型和分詞器
model_name = "distilgpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# 初始輸入文本
input_text = "I love using Hugging Face Transformers!"

# 將文本轉換為模型輸入格式
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# 設置最大生成長度
max_length = 1000  # 最大 tokens 數量
current_length = input_ids.shape[1]  # 當前 tokens 數量

# 開始生成文本
while current_length < max_length:
    # 生成下一個 token
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_length=current_length + 1,  # 每次生成一個 token
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,  # 設置結束符
            do_sample=True,  # 啟用隨機採樣
            top_k=50,  # 限制候選詞數量
            top_p=0.95,  # 使用 nucleus sampling
            temperature=0.7,  # 控制多樣性
        )
    
    # 更新 input_ids 為新生成的文本
    input_ids = outputs
    
    # 解碼生成的文本
    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    
    # 輸出當前生成的文本
    print(generated_text)
    
    # 檢查是否生成結束符
    if tokenizer.eos_token_id in input_ids[0]:
        print("\n生成結束：遇到結束符。")
        break
    
    # 更新當前 tokens 數量
    current_length = input_ids.shape[1]

# 最終生成的文本
print("\n最終生成的文本：")
print(generated_text)