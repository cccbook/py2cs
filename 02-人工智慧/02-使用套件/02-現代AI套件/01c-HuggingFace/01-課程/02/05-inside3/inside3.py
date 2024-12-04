import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
print('tokenizer=', tokenizer)

model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!", "I hate this so much!"]

tokens = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")
print('tokens=', tokens)

output = model(**tokens)
print('output=', output)

import torch

predictions = torch.nn.functional.softmax(output.logits, dim=-1)
print('predictions=\n', predictions)

print('model.config.id2label=', model.config.id2label)

idx = predictions.argmax(axis=1)
print('idx=', idx)

labels = [model.config.id2label[i.item()] for i in idx]
print('labels=', labels)