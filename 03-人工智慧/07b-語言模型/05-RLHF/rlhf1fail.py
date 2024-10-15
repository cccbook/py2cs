import torch
import torch.nn as nn
import torch.optim as optim
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import pipeline

# 設定裝置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 載入 GPT-2 模型和 tokenizer
model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# 獎勵模型（假設已經有一個基於人類反饋訓練的獎勵模型）
reward_model = pipeline('sentiment-analysis')  # 簡化為情感分析

# PPO 的一些超參數
lr = 1e-5  # 學習率
ppo_epochs = 4  # 每次更新的次數
epsilon_clip = 0.2  # PPO clip 範圍

# 定義優化器
optimizer = optim.Adam(model.parameters(), lr=lr)

# 生成文本的函數
def generate_text(prompt, model, tokenizer, max_length=50):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(inputs.input_ids, max_length=max_length, do_sample=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 計算獎勵（基於簡化的情感評分）
def get_reward(text, reward_model):
    result = reward_model(text)[0]
    # 假設正面情感評分越高，獎勵越大
    reward = result['score'] if result['label'] == 'POSITIVE' else -result['score']
    return reward

# PPO 策略更新
def ppo_update(old_log_probs, rewards, states, actions, new_log_probs):
    # 計算優勢函數 A_t (簡化計算)
    advantages = rewards - rewards.mean()

    # 計算 PPO 的目標函數
    ratios = torch.exp(new_log_probs - old_log_probs)
    surr1 = ratios * advantages
    surr2 = torch.clamp(ratios, 1 - epsilon_clip, 1 + epsilon_clip) * advantages
    loss = -torch.min(surr1, surr2).mean()

    # 反向傳播更新策略
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 模擬一個簡化的 RLHF 過程
def train_ppo(prompt, model, tokenizer, reward_model, epochs=10):
    for epoch in range(epochs):
        # 生成文本
        text = generate_text(prompt, model, tokenizer)
        
        # 獲取獎勵
        reward = get_reward(text, reward_model)
        
        # 計算 log_probs（假設動作是生成每個 token 的機率）
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model(**inputs, labels=inputs.input_ids)
        log_probs = outputs.logits.log_softmax(dim=-1)
        
        # 假設當前 log_probs 作為 "舊" 的策略
        old_log_probs = log_probs.clone().detach()
        
        # PPO 策略更新
        for _ in range(ppo_epochs):
            new_text = generate_text(prompt, model, tokenizer)
            new_reward = get_reward(new_text, reward_model)
            new_outputs = model(**inputs, labels=inputs.input_ids)
            new_log_probs = new_outputs.logits.log_softmax(dim=-1)
            
            # 更新策略
            ppo_update(old_log_probs, reward, inputs.input_ids, None, new_log_probs)

        print(f"Epoch {epoch+1}: Text generated: {text}, Reward: {reward}")

# 主程序
prompt = "Once upon a time"
train_ppo(prompt, model, tokenizer, reward_model)
