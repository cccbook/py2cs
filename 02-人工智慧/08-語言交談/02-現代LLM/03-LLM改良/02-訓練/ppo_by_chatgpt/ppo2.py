import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# 假設的 Reward Model，用來評估生成的文本
class RewardModel(nn.Module):
    def __init__(self):
        super(RewardModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, hidden_states):
        # 輸入是 GPT 模型的隱層輸出
        reward = self.fc(hidden_states)
        return reward

# 策略網絡，基於 GPT2
class PolicyNetwork(nn.Module):
    def __init__(self, model_name="gpt2"):
        super(PolicyNetwork, self).__init__()
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    def forward(self, input_ids):
        outputs = self.model(input_ids, output_hidden_states=True)
        logits = outputs.logits  # 輸出 logits，用於生成下一個 token
        hidden_states = outputs.hidden_states[-1]  # 隱層輸出，用於獎勵模型
        return logits, hidden_states

    def generate_text(self, prompt, max_length=50):
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        output = self.model.generate(input_ids, max_length=max_length, do_sample=True)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

# PPO 更新
class PPOAgent:
    def __init__(self, policy_net, reward_model, lr=1e-5, gamma=0.99, clip_epsilon=0.2):
        self.policy_net = policy_net
        self.reward_model = reward_model
        self.optimizer = optim.Adam(self.policy_net.model.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon

    def compute_rewards(self, hidden_states):
        rewards = self.reward_model(hidden_states).squeeze()
        return rewards

    def compute_advantage(self, rewards, values):
        return rewards - values

    def ppo_update(self, old_log_probs, new_log_probs, advantages):
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

    def update(self, input_ids, actions, rewards):
        logits, hidden_states = self.policy_net(input_ids)
        values = self.compute_rewards(hidden_states)

        advantages = self.compute_advantage(rewards, values).detach()

        old_logits = logits.detach()
        old_log_probs = nn.functional.log_softmax(old_logits, dim=-1).gather(1, actions.unsqueeze(1))

        new_logits = logits
        new_log_probs = nn.functional.log_softmax(new_logits, dim=-1).gather(1, actions.unsqueeze(1))

        self.ppo_update(old_log_probs, new_log_probs, advantages)

# 簡化的訓練流程
def train_ppo_rlhf():
    # 初始化策略網絡和獎勵模型
    policy_net = PolicyNetwork()
    reward_model = RewardModel()
    agent = PPOAgent(policy_net, reward_model)

    # 假設的訓練數據
    prompts = ["Explain quantum physics in simple terms.", 
               "Write a poem about the ocean.", 
               "Tell me a joke about computers."]

    for episode in range(1000):
        for prompt in prompts:
            # 生成文本
            generated_text = policy_net.generate_text(prompt)

            # 將生成的文本轉換為 input_ids
            input_ids = policy_net.tokenizer.encode(prompt, return_tensors="pt")
            actions = input_ids[:, -1]  # 假設動作是最後一個生成的 token

            # 獲取模型的隱層輸出
            logits, hidden_states = policy_net(input_ids)

            # 使用獎勵模型評估生成的文本
            rewards = agent.compute_rewards(hidden_states)

            # PPO 更新
            agent.update(input_ids, actions, rewards)

        print(f"Episode {episode} complete. Example generated text: {generated_text}")

if __name__ == "__main__":
    train_ppo_rlhf()
