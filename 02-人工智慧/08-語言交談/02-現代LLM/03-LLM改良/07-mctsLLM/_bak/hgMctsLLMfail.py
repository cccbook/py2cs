import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import math
import random

# 設定 MPS 為運行設備
device = torch.device("mps") if torch.has_mps else torch.device("cpu")

# 載入 GPT-2 模型和 tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# 定義 MCTS 節點
class MCTSNode:
    def __init__(self, token_id, parent=None):
        self.token_id = token_id
        self.parent = parent
        self.children = {}
        self.visit_count = 0
        self.value = 0.0

    def add_child(self, token_id):
        if token_id not in self.children:
            self.children[token_id] = MCTSNode(token_id, self)
        return self.children[token_id]

    def ucb_score(self, exploration_param=1.0):
        if self.visit_count == 0:
            return float('inf')  # 未訪問的節點
        avg_value = self.value / self.visit_count
        exploration = exploration_param * math.sqrt(math.log(self.parent.visit_count) / self.visit_count)
        return avg_value + exploration

# MCTS 搜索過程
def mcts_search(model, tokenizer, root, max_steps=10, max_length=100, exploration_param=1.0):
    for _ in range(max_steps):
        node = root
        generated_tokens = []

        # 1. 選擇：基於 UCB 選擇最佳路徑
        while node.children:
            node = max(node.children.values(), key=lambda child: child.ucb_score(exploration_param))
            generated_tokens.append(node.token_id)

        # 2. 擴展：利用 GPT-2 模型生成新 token
        input_ids = torch.tensor([generated_tokens], device=device)
        with torch.no_grad():
            outputs = model(input_ids)
        logits = outputs.logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        
        # 從概率分佈中選取新 token
        token_id = torch.multinomial(probs, num_samples=1).item()

        # 3. 更新：將新 token 加入子節點並更新訪問計數和評分
        child_node = node.add_child(token_id)
        reward = evaluate_reward(token_id)  # 使用自訂的評估函數
        backpropagate(child_node, reward)

    # 最佳 token 為訪問次數最多的節點
    best_token = max(root.children.values(), key=lambda child: child.visit_count).token_id
    return best_token

# 回溯更新節點
def backpropagate(node, reward):
    while node:
        node.visit_count += 1
        node.value += reward
        node = node.parent

# 獎勵評估函數（可以自訂）
def evaluate_reward(token_id):
    return random.uniform(0, 1)  # 使用隨機值進行演示

# 初始化 MCTS 並生成文本
prompt = "今天的天氣真好，我們來談談人工智慧的未來。"
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
root_node = MCTSNode(input_ids[0, -1].item())  # 初始節點

output_tokens = []
for _ in range(20):  # 每輪生成 20 個 token
    best_token = mcts_search(model, tokenizer, root_node, max_steps=10)
    output_tokens.append(best_token)
    root_node = root_node.add_child(best_token)  # 更新起始節點

# 最終輸出結果
generated_text = tokenizer.decode(output_tokens, skip_special_tokens=True)
print(generated_text)
