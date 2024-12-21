# https://chatgpt.com/c/67287982-904c-8012-bc87-23c3b4e5fa62
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from collections import defaultdict

# 檢查 MPS 是否可用
device = torch.device("mps" if torch.backends.mps.is_built() else "cpu")

# 定義 Trie 節點，用於儲存 token 出現次數
class TrieNode:
    def __init__(self):
        self.children = defaultdict(TrieNode)
        self.count = 0

class Trie:
    def __init__(self):
        self.root = TrieNode()
        
    def insert(self, tokens):
        node = self.root
        for token in tokens:
            node = node.children[token]
            node.count += 1
            
    def get_token_counts(self, tokens):
        node = self.root
        for token in tokens:
            if token in node.children:
                node = node.children[token]
            else:
                return {}
        return {token: child.count for token, child in node.children.items()}

# MCTS 搜尋
class MCTS:
    def __init__(self, model, tokenizer, trie, confidence=1.0):
        self.model = model
        self.tokenizer = tokenizer
        self.trie = trie
        self.confidence = confidence
        
    def select_next_token(self, context_ids):
        # 將 context 轉換為 tensor，並傳至 MPS 計算
        inputs = torch.tensor([context_ids]).to(device)
        with torch.no_grad():
            outputs = self.model(inputs)
        token_probs = torch.softmax(outputs.logits[0, -1, :], dim=-1)
        
        # 使用 Trie 結構中的 token 次數來計算信賴區間上界
        trie_counts = self.trie.get_token_counts(context_ids)
        
        max_score = -float('inf')
        best_token = None
        for token_id, prob in enumerate(token_probs):
            if token_id in trie_counts:
                upper_conf = prob + self.confidence / (1 + trie_counts[token_id])
            else:
                upper_conf = prob + self.confidence
            
            if upper_conf > max_score:
                max_score = upper_conf
                best_token = token_id

        return best_token

# 初始化 GPT-2 模型並移至 MPS
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
trie = Trie()

# 設定初始 context（可以根據實際需要調整）
context_text = "Once upon a time"
context_ids = tokenizer.encode(context_text)

# 使用 MCTS 選擇 token
mcts = MCTS(model, tokenizer, trie, confidence=1.0)
for _ in range(20):  # 模擬生成 20 個 token 的句子
    next_token_id = mcts.select_next_token(context_ids)
    context_ids.append(next_token_id)
    
    # 更新 Trie 結構
    trie.insert(context_ids)
    
    # 將生成的 token 解碼並顯示
    generated_text = tokenizer.decode(context_ids)
    print(generated_text)
