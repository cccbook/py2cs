from datasets import load_dataset

# 加載 IMDb 數據集
dataset = load_dataset('imdb')
train_dataset = dataset['train']
test_dataset = dataset['test']

from transformers import BertTokenizer

# 初始化 BERT 分詞器
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 定義分詞函數
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)

# 對訓練和測試集進行分詞
train_tokenized = train_dataset.map(tokenize_function, batched=True)
test_tokenized = test_dataset.map(tokenize_function, batched=True)


from transformers import BertForSequenceClassification

# 加載預訓練的 BERT 模型，用於序列分類
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)


from transformers import Trainer, TrainingArguments

# 設置訓練參數
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

# 定義訓練器
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=test_tokenized,
)


# 開始訓練
trainer.train()


# 評估模型
results = trainer.evaluate()
print(results)

# 進行預測
predictions = trainer.predict(test_tokenized)
predicted_labels = predictions.predictions.argmax(-1)

# 顯示前幾個預測結果
for i in range(5):
    print(f"Predicted: {predicted_labels[i]}, Actual: {test_dataset[i]['label']}")

"""
RuntimeError: MPS backend out of memory 
(MPS allocated: 8.84 GB, other allocations: 195.75 MB, 
max allowed: 9.07 GB). Tried to allocate 192.00 MB on 
private pool. Use PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 
to disable upper limit for memory allocations (may cause 
system failure).
"""
