import torch
from transformers import pipeline

generator = pipeline(
    "text-generation", 
    model="meta-llama/Llama-3.2-1B", 
    torch_dtype=torch.bfloat16, 
    device_map="auto",
    # early_stopping=False,
    max_length=500,
    no_repeat_ngram_size=2,
    max_new_tokens=300,
)
# 如何控制參數，參考 https://huggingface.co/blog/how-to-generate
# r = generator("In this course, we will teach you how to")
r = generator("問題：請問 相對論 是什麼？ 回答：")
print(r)