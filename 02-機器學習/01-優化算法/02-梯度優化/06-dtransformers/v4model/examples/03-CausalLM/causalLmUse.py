# from transformers import AutoModelForCausalLM, AutoTokenizer
from dtransformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

# 指定 GPT-2 模型名稱
model_name = "gpt2"  # 任何有 generate 方法的模型都可以使用
# model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# 自動載入模型，配置與 tokenizer
config = AutoConfig.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

print('config:', config)
# 設定輸入文本
print('============================')
input_text = "Once upon a time"
inputs = tokenizer(input_text, return_tensors="pt")

# 讓模型生成文字
output_tokens = model.generate(
    **inputs,
    max_length=100,  # 生成最多 100 個 token
    temperature=0.7,  # 控制隨機性，較低的值產生較為穩定的結果
    top_p=0.9,  # Top-p (nucleus sampling)
    # repetition_penalty=1.1,  # 防止重複詞彙
    do_sample=True  # 啟用隨機取樣
)

# 解碼並輸出生成的文本
generated_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
print("生成的文本：")
print(generated_text)
