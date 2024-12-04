from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

sequence = "Using a Transformer network is simple"
tokens = tokenizer.tokenize(sequence)

print('tokens=', tokens)

ids = tokenizer.convert_tokens_to_ids(tokens)

print('ids=', ids)

# 請注意， decode 方法不僅將索引轉換回標記(token)，還將屬於相同單詞的標記(token)組合在一起以生成可讀的句子。
# decoded_string = tokenizer.decode([7993, 170, 11303, 1200, 2443, 1110, 3014])
decoded_string = tokenizer.decode(ids)

print('decoded_string=', decoded_string)
