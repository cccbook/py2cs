from transformers import AutoTokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
print('inputs=', inputs)

from transformers import AutoModel

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModel.from_pretrained(checkpoint)

outputs = model(**inputs)
print('outputs.last_hidden_state.shape=\n', outputs.last_hidden_state.shape)

from transformers import AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
outputs = model(**inputs)

print('outputs.logits.shape=\n', outputs.logits.shape)

print('outputs.logits=\n', outputs.logits)

import torch

predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
print('predictions=\n', predictions)

print('model.config.id2label=', model.config.id2label)

idx = predictions.argmax(axis=1)
print('idx=', idx)

labels = [model.config.id2label[i.item()] for i in idx]
print('labels=', labels)
