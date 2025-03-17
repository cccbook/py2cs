from transformers import GPT2LMHeadModel, GPT2Tokenizer

# 加載分詞器和模型
model_name = "distilgpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# 輸入提示
input_text = "What is the capital of France?"

# 將輸入文本轉換為模型輸入的 token
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# 設置 attention_mask
attention_mask = input_ids.ne(tokenizer.eos_token_id).int()
# 生成文本
output = model.generate(
    input_ids,
    attention_mask=attention_mask,  # 設置 attention_mask
    pad_token_id=tokenizer.eos_token_id,  # 設置 pad_token_id
    max_length=50,  # 生成的最大長度
    num_return_sequences=1,  # 生成的序列數量
    no_repeat_ngram_size=2,  # 避免重複的 n-gram
    # top_p=0.95,  # 使用 nucleus sampling
    # temperature=0.7,  # 控制生成的多樣性
)

# 解碼生成的文本
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(generated_text)