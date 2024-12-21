import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class CharRNN(nn.Module):
    """使用 PyTorch 的字符級語言模型"""
    def __init__(self, vocab_size: int, hidden_size: int = 256):
        super().__init__()
        
        self.hidden_size = hidden_size
        
        # 字符嵌入層
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        
        # RNN 層
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
        
        # 輸出層
        self.fc = nn.Linear(hidden_size, vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(p=0.5)
        
    def forward(self, x: torch.Tensor, h: torch.Tensor = None, training: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        # x shape: (batch_size, sequence_length)
        batch_size, seq_length = x.size()
        
        # 獲取嵌入
        x = self.embedding(x)  # shape: (batch_size, sequence_length, hidden_size)
        
        # 初始化隱藏狀態
        if h is None:
            h = torch.zeros(1, batch_size, self.hidden_size).to(x.device)
        
        # RNN 前向傳播
        x, h = self.rnn(x, h)  # shape: (batch_size, sequence_length, hidden_size)
        
        # 應用 dropout
        if training:
            x = self.dropout(x)
        
        # 應用輸出層
        logits = self.fc(x)  # shape: (batch_size, sequence_length, vocab_size)
        
        return logits, h

def create_sequences(text: str, seq_length: int) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    """創建訓練序列和字符映射"""
    # 創建字符到索引的映射
    chars = sorted(list(set(text)))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    
    # 創建序列
    sequences = []
    for i in range(0, len(text) - seq_length):
        sequences.append(text[i:i + seq_length + 1])  # +1 為了包含目標字符
    
    return sequences, char_to_idx, idx_to_char

def prepare_batch(sequences: List[str], char_to_idx: Dict[str, int], batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """準備一個批次的數據"""
    # 隨機選擇序列
    batch_sequences = np.random.choice(sequences, batch_size)
    
    # 轉換為索引
    X = np.array([[char_to_idx[c] for c in seq[:-1]] for seq in batch_sequences])
    y = np.array([[char_to_idx[c] for c in seq[1:]] for seq in batch_sequences])
    
    return torch.tensor(X, dtype=torch.long), torch.tensor(y, dtype=torch.long)

def loss_fn(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """計算損失"""
    logits, _ = model(X)
    # 重塑 logits 和目標以匹配交叉熵的預期形狀
    logits_2d = logits.reshape(-1, logits.size(-1))  # (batch_size * seq_length, vocab_size)
    targets_1d = y.reshape(-1)  # (batch_size * seq_length,)
    return nn.CrossEntropyLoss()(logits_2d, targets_1d)

def generate_text(
    model: nn.Module,
    seed_text: str,
    char_to_idx: Dict[str, int],
    idx_to_char: Dict[int, str],
    length: int = 100,
    temperature: float = 1.0
) -> str:
    """生成文本"""
    model.eval()
    
    # 初始化
    generated = seed_text
    h = None
    
    # 轉換種子文本為索引
    x = torch.tensor([[char_to_idx[c] for c in seed_text]], dtype=torch.long)
    
    # 生成字符
    for _ in range(length):
        # 獲取預測
        logits, h = model(x, h, training=False)
        logits = logits[:, -1, :] / temperature  # 只使用最後一個時間步
        
        # 計算概率
        probs = torch.softmax(logits, dim=-1).squeeze().tolist()
        
        # 確保概率和為 1
        probs = [p / sum(probs) for p in probs]
        
        # 採樣下一個字符
        next_char_idx = np.random.choice(len(idx_to_char), p=probs)
        
        # 添加到生成的文本
        generated += idx_to_char[next_char_idx]
        
        # 更新輸入
        x = torch.tensor([[next_char_idx]], dtype=torch.long)
    
    return generated

def main(args):
    # 加載文本數據
    with open(args.input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # 創建序列和映射
    sequences, char_to_idx, idx_to_char = create_sequences(text, args.sequence_length)
    vocab_size = len(char_to_idx)
    print(f"Vocabulary size: {vocab_size}")
    
    # 創建模型
    model = CharRNN(vocab_size, args.hidden_size)
    if args.gpu:
        model = model.cuda()
    
    # 設置優化器
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # 訓練循環
    for epoch in range(args.num_epochs):
        tic = time.perf_counter()
        total_loss = 0
        num_batches = args.batches_per_epoch
        
        for _ in range(num_batches):
            # 準備批次
            X, y = prepare_batch(sequences, char_to_idx, args.batch_size)
            if args.gpu:
                X, y = X.cuda(), y.cuda()
            
            # 前向傳播
            loss = loss_fn(model, X, y)
            
            # 反向傳播和更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # 計算平均損失
        avg_loss = total_loss / num_batches
        toc = time.perf_counter()
        
        print(f"Epoch {epoch + 1}: avg_loss = {avg_loss:.4f}, time = {toc - tic:.2f}s")
        
        # 生成示例文本
        if (epoch + 1) % args.generate_every == 0:
            seed = text[:args.sequence_length]
            generated = generate_text(
                model,
                seed,
                char_to_idx,
                idx_to_char,
                length=100,
                temperature=args.temperature
            )
            print(f"\nGenerated text:\n{generated}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a character-level RNN")
    parser.add_argument("--input-file", type=str, required=True, help="Input text file")
    parser.add_argument("--sequence-length", type=int, default=50, help="Sequence length")
    parser.add_argument("--hidden-size", type=int, default=256, help="Hidden size")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--num-epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batches-per-epoch", type=int, default=100, help="Batches per epoch")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--generate-every", type=int, default=5, help="Generate text every N epochs")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--gpu", action="store_true", help="Use GPU")
    
    args = parser.parse_args()
    main(args)
