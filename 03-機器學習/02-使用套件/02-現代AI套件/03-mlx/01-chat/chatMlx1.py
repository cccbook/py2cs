# https://huggingface.co/mlx-community/Llama-3.2-3B-bf16

from mlx_lm import load, generate

model, tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-4bit")
response = generate(model, tokenizer, prompt="什麼是 Llama AI 模型？", verbose=True)
