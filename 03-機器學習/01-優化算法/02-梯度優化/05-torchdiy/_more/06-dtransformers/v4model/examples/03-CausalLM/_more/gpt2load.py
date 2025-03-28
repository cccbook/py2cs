from transformers import GPT2LMHeadModel, GPT2Config
import torch
import json
import os

# 讀取 config.json
with open("gpt2_model/config.json", "r") as f:
    config_data = json.load(f)
config = GPT2Config.from_dict(config_data)

# 建立 GPT-2 模型
model = GPT2LMHeadModel(config)

# 讀取權重
state_dict = torch.load("gpt2_model/gpt2lmhead.pt")
model.load_state_dict(state_dict)

print("GPT-2 模型已成功載入！")
