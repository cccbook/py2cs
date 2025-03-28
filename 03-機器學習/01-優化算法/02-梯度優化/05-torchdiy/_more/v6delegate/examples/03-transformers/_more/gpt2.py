from transformers import GPT2Tokenizer, GPT2LMHeadModel

# 加載預訓練的 GPT-2 模型和對應的分詞器
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# 輸入問題
question = "What is the capital of France?"

# 將問題轉換為模型輸入格式
input_ids = tokenizer.encode(question, return_tensors="pt")

# 生成回答
# 設置 max_length 控制生成文本的長度，num_return_sequences 控制生成多少個回答
output = model.generate(
    input_ids,
    max_length=50,  # 生成文本的最大長度
    num_return_sequences=1,  # 生成一個回答
    no_repeat_ngram_size=2,  # 避免重複的 n-gram
    top_k=50,  # 限制候選詞的數量
    top_p=0.95,  # 使用 nucleus sampling
    temperature=0.7,  # 控制生成文本的多樣性
)

# 解碼生成的文本
answer = tokenizer.decode(output[0], skip_special_tokens=True)

# 輸出結果
print(f"Question: {question}")
print(f"Answer: {answer}")