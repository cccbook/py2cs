from transformers import GPT2LMHeadModel, GPT2Config
import torch
import json
import os

# 指定 Hugging Face 的模型名稱
model_name = "gpt2"

# 下載並載入 GPT-2 模型和其設定
config = GPT2Config.from_pretrained(model_name)  # 取得 config.json 的內容
model = GPT2LMHeadModel.from_pretrained(model_name, config=config)  # 取得權重

# 指定儲存目錄
save_dir = "gpt2_model"
os.makedirs(save_dir, exist_ok=True)

# 儲存 config.json
config_path = os.path.join(save_dir, "config.json")
with open(config_path, "w") as f:
    json.dump(config.to_dict(), f, indent=4)

# 儲存 PyTorch 權重為 gpt2.pt
weights_path = os.path.join(save_dir, "gpt2.pt")
torch.save(model.state_dict(), weights_path)

print(f"模型已儲存至 {save_dir}")
print(f"- Config: {config_path}")
print(f"- Weights: {weights_path}")
