import mlx.core as mx
from model import Bert, load_model

model, tokenizer = load_model(
    "bert-base-uncased",
    "weights/bert-base-uncased.npz")

batch = ["This is an example of BERT working on MLX."]
tokens = tokenizer(batch, return_tensors="np", padding=True)
tokens = {key: mx.array(v) for key, v in tokens.items()}

output, pooled = model(**tokens)
print('output=', output)
print('pooled=', pooled)
