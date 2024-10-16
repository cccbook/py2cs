from pathlib import Path
import tiktoken
from tiktoken.load import load_tiktoken_bpe
import torch
import json
import matplotlib.pyplot as plt

tokenizer_path = "Meta-Llama-3-8B/tokenizer.model"
special_tokens = [
            "<|begin_of_text|>",
            "<|end_of_text|>",
            "<|reserved_special_token_0|>",
            "<|reserved_special_token_1|>",
            "<|reserved_special_token_2|>",
            "<|reserved_special_token_3|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|reserved_special_token_4|>",
            "<|eot_id|>",  # end of turn
        ] + [f"<|reserved_special_token_{i}|>" for i in range(5, 256 - 5)]
mergeable_ranks = load_tiktoken_bpe(tokenizer_path)
tokenizer = tiktoken.Encoding(
    name=Path(tokenizer_path).name,
    pat_str=r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
    mergeable_ranks=mergeable_ranks,
    special_tokens={token: len(mergeable_ranks) + i for i, token in enumerate(special_tokens)},
)

r = tokenizer.decode(tokenizer.encode("hello world!"))
print('decode("hello world!")=', r)

model = torch.load("Meta-Llama-3-8B/consolidated.00.pth")
print(json.dumps(list(model.keys())[:20], indent=4))

with open("Meta-Llama-3-8B/params.json", "r") as f:
    config = json.load(f)
print(config)

dim = config["dim"]
n_layers = config["n_layers"]
n_heads = config["n_heads"]
n_kv_heads = config["n_kv_heads"]
vocab_size = config["vocab_size"]
multiple_of = config["multiple_of"]
ffn_dim_multiplier = config["ffn_dim_multiplier"]
norm_eps = config["norm_eps"]
rope_theta = torch.tensor(config["rope_theta"])

prompt = "the answer to the ultimate question of life, the universe, and everything is "
tokens = [128000] + tokenizer.encode(prompt)
print(tokens)
tokens = torch.tensor(tokens)
prompt_split_as_tokens = [tokenizer.decode([token.item()]) for token in tokens]
print(prompt_split_as_tokens)

embedding_layer = torch.nn.Embedding(vocab_size, dim)
embedding_layer.weight.data.copy_(model["tok_embeddings.weight"])
token_embeddings_unnormalized = embedding_layer(tokens).to(torch.bfloat16)
print(token_embeddings_unnormalized.shape)

# def rms_norm(tensor, norm_weights):
#     rms = (tensor.pow(2).mean(-1, keepdim=True) + norm_eps)**0.5
#     return tensor * (norm_weights / rms)
def rms_norm(tensor, norm_weights):
    return (tensor * torch.rsqrt(tensor.pow(2).mean(-1, keepdim=True) + norm_eps)) * norm_weights

token_embeddings = rms_norm(token_embeddings_unnormalized, model["layers.0.attention_norm.weight"])
print('token_embeddings.shape=', token_embeddings.shape)

print(
    'q.shape', model["layers.0.attention.wq.weight"].shape,
    'k.shape', model["layers.0.attention.wk.weight"].shape,
    'v.shape', model["layers.0.attention.wv.weight"].shape,
    'o.shape', model["layers.0.attention.wo.weight"].shape
)

q_layer0 = model["layers.0.attention.wq.weight"]
head_dim = q_layer0.shape[0] // n_heads
q_layer0 = q_layer0.view(n_heads, head_dim, dim)
print('q_layer0.shape=', q_layer0.shape)

q_layer0_head0 = q_layer0[0]
print('q_layer0_head0.shape', q_layer0_head0.shape) # torch.Size([128, 4096])

q_per_token = torch.matmul(token_embeddings, q_layer0_head0.T)
print('q_per_token.shape=', q_per_token.shape) # torch.Size([17, 128])

q_per_token_split_into_pairs = q_per_token.float().view(q_per_token.shape[0], -1, 2)
print('q_per_token_split_into_pairs.shape=', q_per_token_split_into_pairs.shape) # torch.Size([17, 64, 2])

zero_to_one_split_into_64_parts = torch.tensor(range(64))/64
print('zero_to_one_split_into_64_parts=\n', zero_to_one_split_into_64_parts)

freqs = 1.0 / (rope_theta ** zero_to_one_split_into_64_parts)
print('freqs=', freqs)

freqs_for_each_token = torch.outer(torch.arange(17), freqs)
freqs_cis = torch.polar(torch.ones_like(freqs_for_each_token), freqs_for_each_token)
print('freqs_cis.shape=', freqs_cis.shape)

# viewing tjhe third row of freqs_cis
value = freqs_cis[3]
plt.figure()
for i, element in enumerate(value[:17]):
    plt.plot([0, element.real], [0, element.imag], color='blue', linewidth=1, label=f"Index: {i}")
    plt.annotate(f"{i}", xy=(element.real, element.imag), color='red')
plt.xlabel('Real')
plt.ylabel('Imaginary')
plt.title('Plot of one row of freqs_cis')
plt.show()

q_per_token_as_complex_numbers = torch.view_as_complex(q_per_token_split_into_pairs)
print('q_per_token_as_complex_numbers.shape=', q_per_token_as_complex_numbers.shape) # torch.Size([17, 64])

q_per_token_as_complex_numbers_rotated = q_per_token_as_complex_numbers * freqs_cis
print('q_per_token_as_complex_numbers_rotated.shape=', q_per_token_as_complex_numbers_rotated.shape) # torch.Size([17, 64])

q_per_token_split_into_pairs_rotated = torch.view_as_real(q_per_token_as_complex_numbers_rotated)
print('q_per_token_split_into_pairs_rotated.shape=', q_per_token_split_into_pairs_rotated.shape) # torch.Size([17, 64, 2])

q_per_token_rotated = q_per_token_split_into_pairs_rotated.view(q_per_token.shape)
print('q_per_token_rotated.shape=', q_per_token_rotated.shape) # torch.Size([17, 128])

k_layer0 = model["layers.0.attention.wk.weight"]
k_layer0 = k_layer0.view(n_kv_heads, k_layer0.shape[0] // n_kv_heads, dim)
print('k_layer0.shape=', k_layer0.shape) # torch.Size([8, 128, 4096])

k_layer0_head0 = k_layer0[0]
print('k_layer0_head0.shape=', k_layer0_head0.shape) # torch.Size([128, 4096])

k_per_token = torch.matmul(token_embeddings, k_layer0_head0.T)
print('k_per_token.shape=', k_per_token.shape) # torch.Size([17, 128])

k_per_token_split_into_pairs = k_per_token.float().view(k_per_token.shape[0], -1, 2)
print('k_per_token_split_into_pairs.shape=', k_per_token_split_into_pairs.shape) # torch.Size([17, 64, 2])

k_per_token_as_complex_numbers = torch.view_as_complex(k_per_token_split_into_pairs)
print('k_per_token_as_complex_numbers.shape=', k_per_token_as_complex_numbers.shape) # torch.Size([17, 64])

k_per_token_split_into_pairs_rotated = torch.view_as_real(k_per_token_as_complex_numbers * freqs_cis)
# k_per_token_split_into_pairs_rotated.shape

k_per_token_split_into_pairs_rotated = torch.view_as_real(k_per_token_as_complex_numbers * freqs_cis)
# k_per_token_split_into_pairs_rotated.shape

k_per_token_rotated = k_per_token_split_into_pairs_rotated.view(k_per_token.shape)
# k_per_token_rotated.shape

qk_per_token = torch.matmul(q_per_token_rotated, k_per_token_rotated.T)/(head_dim)**0.5
# qk_per_token.shape

def display_qk_heatmap(qk_per_token):
    _, ax = plt.subplots()
    im = ax.imshow(qk_per_token.to(float).detach(), cmap='viridis')
    ax.set_xticks(range(len(prompt_split_as_tokens)))
    ax.set_yticks(range(len(prompt_split_as_tokens)))
    ax.set_xticklabels(prompt_split_as_tokens)
    ax.set_yticklabels(prompt_split_as_tokens)
    ax.figure.colorbar(im, ax=ax)
    
display_qk_heatmap(qk_per_token)

mask = torch.full((len(tokens), len(tokens)), float("-inf"), device=tokens.device)
mask = torch.triu(mask, diagonal=1)
# mask

qk_per_token_after_masking = qk_per_token + mask
display_qk_heatmap(qk_per_token_after_masking)

qk_per_token_after_masking_after_softmax = torch.nn.functional.softmax(qk_per_token_after_masking, dim=1).to(torch.bfloat16)
display_qk_heatmap(qk_per_token_after_masking_after_softmax)

v_layer0 = model["layers.0.attention.wv.weight"]
v_layer0 = v_layer0.view(n_kv_heads, v_layer0.shape[0] // n_kv_heads, dim)
# v_layer0.shape

v_layer0_head0 = v_layer0[0]
# v_layer0_head0.shape

v_per_token = torch.matmul(token_embeddings, v_layer0_head0.T)
# v_per_token.shape

qkv_attention = torch.matmul(qk_per_token_after_masking_after_softmax, v_per_token)
# qkv_attention.shape

qkv_attention_store = []

for head in range(n_heads):
    q_layer0_head = q_layer0[head]
    k_layer0_head = k_layer0[head//4] # key weights are shared across 4 heads
    v_layer0_head = v_layer0[head//4] # value weights are shared across 4 heads
    q_per_token = torch.matmul(token_embeddings, q_layer0_head.T)
    k_per_token = torch.matmul(token_embeddings, k_layer0_head.T)
    v_per_token = torch.matmul(token_embeddings, v_layer0_head.T)

    q_per_token_split_into_pairs = q_per_token.float().view(q_per_token.shape[0], -1, 2)
    q_per_token_as_complex_numbers = torch.view_as_complex(q_per_token_split_into_pairs)
    q_per_token_split_into_pairs_rotated = torch.view_as_real(q_per_token_as_complex_numbers * freqs_cis[:len(tokens)])
    q_per_token_rotated = q_per_token_split_into_pairs_rotated.view(q_per_token.shape)

    k_per_token_split_into_pairs = k_per_token.float().view(k_per_token.shape[0], -1, 2)
    k_per_token_as_complex_numbers = torch.view_as_complex(k_per_token_split_into_pairs)
    k_per_token_split_into_pairs_rotated = torch.view_as_real(k_per_token_as_complex_numbers * freqs_cis[:len(tokens)])
    k_per_token_rotated = k_per_token_split_into_pairs_rotated.view(k_per_token.shape)

    qk_per_token = torch.matmul(q_per_token_rotated, k_per_token_rotated.T)/(128)**0.5
    mask = torch.full((len(tokens), len(tokens)), float("-inf"), device=tokens.device)
    mask = torch.triu(mask, diagonal=1)
    qk_per_token_after_masking = qk_per_token + mask
    qk_per_token_after_masking_after_softmax = torch.nn.functional.softmax(qk_per_token_after_masking, dim=1).to(torch.bfloat16)
    qkv_attention = torch.matmul(qk_per_token_after_masking_after_softmax, v_per_token)
    qkv_attention = torch.matmul(qk_per_token_after_masking_after_softmax, v_per_token)
    qkv_attention_store.append(qkv_attention)

# len(qkv_attention_store)

stacked_qkv_attention = torch.cat(qkv_attention_store, dim=-1)
# stacked_qkv_attention.shape

w_layer0 = model["layers.0.attention.wo.weight"]
# w_layer0.shape

embedding_delta = torch.matmul(stacked_qkv_attention, w_layer0.T)
# embedding_delta.shape

embedding_after_edit = token_embeddings_unnormalized + embedding_delta
# embedding_after_edit.shape

embedding_after_edit_normalized = rms_norm(embedding_after_edit, model["layers.0.ffn_norm.weight"])
# embedding_after_edit_normalized.shape

w1 = model["layers.0.feed_forward.w1.weight"]
w2 = model["layers.0.feed_forward.w2.weight"]
w3 = model["layers.0.feed_forward.w3.weight"]
output_after_feedforward = torch.matmul(torch.functional.F.silu(torch.matmul(embedding_after_edit_normalized, w1.T)) * torch.matmul(embedding_after_edit_normalized, w3.T), w2.T)
# output_after_feedforward.shape

layer_0_embedding = embedding_after_edit+output_after_feedforward
# layer_0_embedding.shape

final_embedding = token_embeddings_unnormalized
for layer in range(n_layers):
    qkv_attention_store = []
    layer_embedding_norm = rms_norm(final_embedding, model[f"layers.{layer}.attention_norm.weight"])
    q_layer = model[f"layers.{layer}.attention.wq.weight"]
    q_layer = q_layer.view(n_heads, q_layer.shape[0] // n_heads, dim)
    k_layer = model[f"layers.{layer}.attention.wk.weight"]
    k_layer = k_layer.view(n_kv_heads, k_layer.shape[0] // n_kv_heads, dim)
    v_layer = model[f"layers.{layer}.attention.wv.weight"]
    v_layer = v_layer.view(n_kv_heads, v_layer.shape[0] // n_kv_heads, dim)
    w_layer = model[f"layers.{layer}.attention.wo.weight"]
    for head in range(n_heads):
        q_layer_head = q_layer[head]
        k_layer_head = k_layer[head//4]
        v_layer_head = v_layer[head//4]
        q_per_token = torch.matmul(layer_embedding_norm, q_layer_head.T)
        k_per_token = torch.matmul(layer_embedding_norm, k_layer_head.T)
        v_per_token = torch.matmul(layer_embedding_norm, v_layer_head.T)
        q_per_token_split_into_pairs = q_per_token.float().view(q_per_token.shape[0], -1, 2)
        q_per_token_as_complex_numbers = torch.view_as_complex(q_per_token_split_into_pairs)
        q_per_token_split_into_pairs_rotated = torch.view_as_real(q_per_token_as_complex_numbers * freqs_cis)
        q_per_token_rotated = q_per_token_split_into_pairs_rotated.view(q_per_token.shape)
        k_per_token_split_into_pairs = k_per_token.float().view(k_per_token.shape[0], -1, 2)
        k_per_token_as_complex_numbers = torch.view_as_complex(k_per_token_split_into_pairs)
        k_per_token_split_into_pairs_rotated = torch.view_as_real(k_per_token_as_complex_numbers * freqs_cis)
        k_per_token_rotated = k_per_token_split_into_pairs_rotated.view(k_per_token.shape)
        qk_per_token = torch.matmul(q_per_token_rotated, k_per_token_rotated.T)/(128)**0.5
        mask = torch.full((len(token_embeddings_unnormalized), len(token_embeddings_unnormalized)), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        qk_per_token_after_masking = qk_per_token + mask
        qk_per_token_after_masking_after_softmax = torch.nn.functional.softmax(qk_per_token_after_masking, dim=1).to(torch.bfloat16)
        qkv_attention = torch.matmul(qk_per_token_after_masking_after_softmax, v_per_token)
        qkv_attention_store.append(qkv_attention)

    stacked_qkv_attention = torch.cat(qkv_attention_store, dim=-1)
    w_layer = model[f"layers.{layer}.attention.wo.weight"]
    embedding_delta = torch.matmul(stacked_qkv_attention, w_layer.T)
    embedding_after_edit = final_embedding + embedding_delta
    embedding_after_edit_normalized = rms_norm(embedding_after_edit, model[f"layers.{layer}.ffn_norm.weight"])
    w1 = model[f"layers.{layer}.feed_forward.w1.weight"]
    w2 = model[f"layers.{layer}.feed_forward.w2.weight"]
    w3 = model[f"layers.{layer}.feed_forward.w3.weight"]
    output_after_feedforward = torch.matmul(torch.functional.F.silu(torch.matmul(embedding_after_edit_normalized, w1.T)) * torch.matmul(embedding_after_edit_normalized, w3.T), w2.T)
    final_embedding = embedding_after_edit+output_after_feedforward

final_embedding = rms_norm(final_embedding, model["norm.weight"])
# final_embedding.shape

model["output.weight"].shape

logits = torch.matmul(final_embedding[-1], model["output.weight"].T)
# logits.shape

next_token = torch.argmax(logits, dim=-1)
# next_token

next_token_decode = tokenizer.decode([next_token.item()])
print('next_token_decode=', next_token_decode)


# 答案的這個梗 42 來自這本書
# https://www.amazon.com/42-ultimate-question-universe-everything/dp/B0BJTPH9W3












