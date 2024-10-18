# https://huggingface.co/gpt2
from transformers import pipeline, set_seed
generator = pipeline('text-generation', model='gpt2')
set_seed(42)
result = generator("Hello, I'm a language model,", max_length=30, num_return_sequences=5)
print(result)
