from mlx_lm import load, generate

model, tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-4bit")
# model, tokenizer = load("mlx-community/Mistral-7B-Instruct-v0.3-4bit")
prompt = "Write a story about Einstein"
# prompt = "寫一篇愛因斯坦的故事"

messages = [{"role": "user", "content": prompt}]
prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

text = generate(model, tokenizer, prompt=prompt, verbose=True, 
     max_tokens=1000, temp=0.5) # temp 是溫度，預設是 0，每次都一樣。 0.5 會有比較多變化。
