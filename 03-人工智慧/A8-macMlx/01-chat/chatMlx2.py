# https://pypi.org/project/mlx-lm/
from mlx_lm import load, generate

model, tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-4bit")

prompt = "Write a story about Einstein"

messages = [{"role": "user", "content": prompt}]
prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

response = generate(model, tokenizer, prompt=prompt, verbose=True)
