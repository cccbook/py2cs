# Copyright © 2024 Apple Inc.
# https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/examples/chat.py
"""
An example of a multi-turn chat with prompt caching.
"""

from mlx_lm import generate, load
from mlx_lm.models.cache import load_prompt_cache, make_prompt_cache, save_prompt_cache

model, tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-4bit")

# Make the initial prompt cache for the model
prompt_cache = make_prompt_cache(model)

# User turn
prompt = "Hi my name is <Name>."
messages = [{"role": "user", "content": prompt}]
prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

# Assistant response
response = generate(
    model,
    tokenizer,
    prompt=prompt,
    verbose=True,
    temp=0.0,
    prompt_cache=prompt_cache,
)

# User turn
prompt = "What's my name?"
messages = [{"role": "user", "content": prompt}]
prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

# Assistant response
response = generate(
    model,
    tokenizer,
    prompt=prompt,
    verbose=True,
    temp=0.0,
    prompt_cache=prompt_cache,
)

# Save the prompt cache to disk to reuse it at a later time
save_prompt_cache("mistral_prompt.safetensors", prompt_cache)

# Load the prompt cache from disk
prompt_cache = load_prompt_cache("mistral_prompt.safetensors")