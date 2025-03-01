from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# 加載 MiniLM 模型和分詞器
model_name = "microsoft/MiniLM-L12-H384-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)  # 二分類任務

# 輸入文本
text = "I love using Hugging Face Transformers!"

# 將文本轉換為模型輸入格式
inputs = tokenizer(text, return_tensors="pt")

# 模型推理
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# 獲取預測結果
predicted_class = torch.argmax(logits, dim=1).item()
print(f"Predicted class: {predicted_class}")