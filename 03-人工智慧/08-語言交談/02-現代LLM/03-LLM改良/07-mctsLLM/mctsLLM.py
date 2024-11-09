# 劉立行用 Claude 後修改 -- https://gist.github.com/austin362667/762acb712abeba8d425329b4bf0da55b
import numpy as np
import torch
from typing import List, Dict, Optional
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM

class HuggingFaceLLM:
    def __init__(self, model_name: str = "HuggingFaceTB/SmolLM2-135M-Instruct", device: str = 'mps'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, device_map=device)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device)
        self.device = device

    def get_logits(self, context_tokens: List[int]) -> np.ndarray:
        """Get logits for next token prediction"""
        input_ids = torch.tensor([context_tokens]).to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits[:, -1, :].cpu().numpy()[0]
        return logits

    def sample_token(self, logits: np.ndarray, temperature: float = 1.0) -> int:
        """Sample next token from logits"""
        if temperature == 0:
            return int(np.argmax(logits))
        
        probs = np.exp(logits / temperature)
        probs = probs / np.sum(probs)
        return int(np.random.choice(len(probs), p=probs))

@dataclass
class MCTSNode:
    token_id: int
    parent: Optional['MCTSNode']
    children: Dict[int, 'MCTSNode']
    visits: int = 0
    total_value: float = 0.0
    prior_probability: float = 0.0
    
    @property
    def value(self) -> float:
        return self.total_value / (self.visits + 1e-8)

    def ucb_score(self, exploration_constant: float = 1.0) -> float:
        if self.parent is None:
            return 0.0
            
        exploitation = self.value
        exploration = exploration_constant * self.prior_probability * \
            np.sqrt(self.parent.visits) / (1 + self.visits)
            
        return exploitation + exploration

class MCTSLLM:
    def __init__(self, model_name: str = "HuggingFaceTB/SmolLM2-135M-Instruct", 
                 device: str = 'mps', exploration_constant: float = 1.0):
        self.llm = HuggingFaceLLM(model_name, device)
        self.exploration_constant = exploration_constant
        self.root = MCTSNode(token_id=-1, parent=None, children={})

    def select(self, node: MCTSNode) -> List[MCTSNode]:
        path = []
        while node.children:
            node = max(node.children.values(), 
                      key=lambda n: n.ucb_score(self.exploration_constant))
            path.append(node)
        return path
    
    def expand(self, leaf: MCTSNode, context_tokens: List[int]) -> MCTSNode:
        logits = self.llm.get_logits(context_tokens)
        probs = np.exp(logits) / np.sum(np.exp(logits))
        
        new_token = self.llm.sample_token(logits)
        
        if new_token not in leaf.children:
            leaf.children[new_token] = MCTSNode(
                token_id=new_token,
                parent=leaf,
                children={},
                prior_probability=probs[new_token]
            )
            
        return leaf.children[new_token]
    
    def simulate(self, node: MCTSNode, context_tokens: List[int]) -> float:
        current_tokens = context_tokens + [node.token_id]
        
        value = 0.0
        for _ in range(3):
            logits = self.llm.get_logits(current_tokens)
            next_token = self.llm.sample_token(logits)
            current_tokens.append(next_token)
            
            probs = np.exp(logits) / np.sum(np.exp(logits))
            value += np.max(probs)
            
        return value / 3.0
    
    def backpropagate(self, path: List[MCTSNode], value: float):
        for node in reversed(path):
            node.visits += 1
            node.total_value += value
    
    def search(self, context_tokens: List[int], n_iterations: int = 10) -> int:
        for _ in range(n_iterations):
            path = self.select(self.root)
            leaf = path[-1] if path else self.root
            
            child = self.expand(leaf, context_tokens)
            path.append(child)
            
            value = self.simulate(child, context_tokens)
            
            self.backpropagate(path, value)
        
        best_child = max(self.root.children.values(), key=lambda n: n.visits)
        return best_child.token_id

    def generate_text(self, prompt: str, max_tokens: int = 100) -> str:
        """Generate text from a prompt using MCTS-guided sampling"""
        initial_tokens = self.llm.tokenizer.encode(prompt)
        current_tokens = initial_tokens.copy()
        
        for _ in range(max_tokens):
            next_token = self.search(current_tokens)
            current_tokens.append(next_token)
            
            if next_token == self.llm.tokenizer.eos_token_id:
                break
                
            self.root = MCTSNode(token_id=-1, parent=None, children={})
            
        return self.llm.tokenizer.decode(current_tokens)
    
    def generate_text_streaming(self, prompt: str, max_tokens: int = 100):
        """Generate text from a prompt using MCTS-guided sampling, streaming each token as it's generated."""
        initial_tokens = self.llm.tokenizer.encode(prompt)
        current_tokens = initial_tokens.copy()
        
        for _ in range(max_tokens):
            next_token = self.search(current_tokens)
            token_text = self.llm.tokenizer.decode(next_token)
            
            yield token_text
            current_tokens.append(next_token)
            
            if next_token == self.llm.tokenizer.eos_token_id:
                break
                
            self.root = MCTSNode(token_id=-1, parent=None, children={})
        
        # Optional: Final yield for full generated text if desired
        yield self.llm.tokenizer.decode(current_tokens)


def print_trie(node: MCTSNode, tokenizer, depth: int = 0):
    """Recursively print the structure of the MCTS tree (trie)"""
    indent = "  " * depth
    token_text = tokenizer.decode([node.token_id]) if node.token_id != -1 else "<root>"
    print(f"{indent}Token: {token_text} | ID: {node.token_id} | Visits: {node.visits} | Value: {node.value:.4f}")

    for child in node.children.values():
        print_trie(child, tokenizer, depth + 1)

mcts = MCTSLLM(
    model_name="HuggingFaceTB/SmolLM2-135M-Instruct",
    device='mps',
    exploration_constant=1.0
)

print_trie(mcts.root, mcts.llm.tokenizer) 

prompt = "Paris is the capital of"
for e in range(10):
    print(mcts.generate_text(prompt, max_tokens=10))