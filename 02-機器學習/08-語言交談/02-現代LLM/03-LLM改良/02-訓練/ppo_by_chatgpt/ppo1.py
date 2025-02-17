import torch
import torch.nn as nn
import torch.optim as optim
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
import numpy as np
import random

# 設定隨機種子
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# 設定設備
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加載 GPT-2 Tokenizer 和模型
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
config = GPT2Config.from_pretrained('gpt2')
config.pad_token_id = tokenizer.eos_token_id  # 設置 pad_token_id
policy_model = GPT2LMHeadModel.from_pretrained('gpt2', config=config).to(device)
policy_model.train()

# 複製一個舊的策略模型，用於計算比值
old_policy_model = GPT2LMHeadModel.from_pretrained('gpt2', config=config).to(device)
old_policy_model.eval()

# 簡化的獎勵模型，這裡用一個假的獎勵函數來模擬
def reward_model(prompt, response):
    # 這裡我們假設獎勵是根據回應的長度和某些關鍵詞出現與否來決定
    reward = 0
    if "good" in response:
        reward += 1
    if "bad" in response:
        reward -= 1
    reward += len(response.split()) * 0.1  # 獎勵較長的回應
    return reward

# PPO 參數
optimizer = optim.Adam(policy_model.parameters(), lr=1e-5)
clip_epsilon = 0.2
epochs = 3
batch_size = 2
update_steps = 5

# 範例數據
prompts = [
    "Tell me a joke about computers.",
    "How is the weather today?",
    "What is the capital of France?",
    "Give me some advice on staying healthy."
]

# 訓練循環
for epoch in range(epochs):
    for prompt in prompts:
        # 準備輸入
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        
        # 使用策略模型生成回應
        output = policy_model.generate(
            input_ids,
            max_length=50,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id
        )
        response_ids = output[0][input_ids.shape[1]:]  # 獲取生成的回應部分
        response_text = tokenizer.decode(response_ids, skip_special_tokens=True)

        # 計算獎勵
        reward = reward_model(prompt, response_text)
        
        # 計算舊策略下的對數概率
        with torch.no_grad():
            old_logits = old_policy_model(input_ids).logits
            old_log_probs = nn.functional.log_softmax(old_logits, dim=-1)
            old_action_log_probs = old_log_probs.gather(2, output.unsqueeze(-1)).squeeze(-1)
            old_action_log_prob = old_action_log_probs.sum()

        # 計算當前策略下的對數概率
        logits = policy_model(input_ids).logits
        log_probs = nn.functional.log_softmax(logits, dim=-1)
        action_log_probs = log_probs.gather(2, output.unsqueeze(-1)).squeeze(-1)
        action_log_prob = action_log_probs.sum()

        # 計算比值 r(θ)
        ratio = torch.exp(action_log_prob - old_action_log_prob.detach())

        # 計算損失函數
        advantage = reward  # 簡化的優勢，實際中應該減去值函數的估計
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantage
        policy_loss = -torch.min(surr1, surr2)

        # 反向傳播和參數更新
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

    # 更新舊策略模型的參數
    old_policy_model.load_state_dict(policy_model.state_dict())

    print(f"Epoch {epoch + 1} completed.")

# 測試模型
policy_model.eval()
for prompt in prompts:
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    output = policy_model.generate(
        input_ids,
        max_length=50,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id
    )
    response_text = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
    print(f"Prompt: {prompt}\nResponse: {response_text}\n")
