from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import os

# Hugging Face 模型名稱
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# 下載模型與 Tokenizer
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 指定儲存目錄
save_dir = "deepseek_qwen_model"
os.makedirs(save_dir, exist_ok=True)

# 儲存模型配置 (config.json -> model.json)
config_path = os.path.join(save_dir, "model.json")
with open(config_path, "w") as f:
    json.dump(model.config.to_dict(), f, indent=4)

# 儲存 PyTorch 權重 (state_dict)
weights_path = os.path.join(save_dir, "deepseek1.5b.pt")
torch.save(model.state_dict(), weights_path)

print(f"模型已儲存至 {save_dir}")
print(f"- 配置檔案: {config_path}")
print(f"- 權重檔案: {weights_path}")
