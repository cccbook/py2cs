import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import numpy as np
from torch.utils.data import DataLoader, Dataset

class CharRNN(pl.LightningModule):
    """使用 PyTorch Lightning 的字符級語言模型"""
    def __init__(self, vocab_size: int, hidden_size: int = 256, lr: float = 0.001):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.lr = lr
        
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
    
    def training_step(self, batch, batch_idx):
        X, y = batch
        logits, _ = self(X)
        # 重塑 logits 和目標以匹配交叉熵的預期形狀
        logits_2d = logits.reshape(-1, logits.size(-1))  # (batch_size * seq_length, vocab_size)
        targets_1d = y.reshape(-1)  # (batch_size * seq_length,)
        loss = nn.CrossEntropyLoss()(logits_2d, targets_1d)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

class CharDataset(Dataset):
    """字符級語言模型數據集"""
    def __init__(self, text: str, seq_length: int, char_to_idx: Dict[str, int]):
        self.text = text
        self.seq_length = seq_length
        self.char_to_idx = char_to_idx
        self.sequences = self.create_sequences(text)

    def create_sequences(self, text: str) -> List[str]:
        """創建訓練序列"""
        sequences = []
        for i in range(0, len(text) - self.seq_length):
            sequences.append(text[i:i + self.seq_length + 1])  # +1 為了包含目標字符
        return sequences
    
    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        X = np.array([self.char_to_idx[c] for c in seq[:-1]])
        y = np.array([self.char_to_idx[c] for c in seq[1:]])
        return torch.tensor(X, dtype=torch.long), torch.tensor(y, dtype=torch.long)

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
    
    # 創建字符到索引的映射
    chars = sorted(list(set(text)))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    vocab_size = len(char_to_idx)
    print(f"Vocabulary size: {vocab_size}")
    
    # 創建數據集和數據加載器
    dataset = CharDataset(text, args.sequence_length, char_to_idx)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # 創建模型
    model = CharRNN(vocab_size, args.hidden_size, args.learning_rate)
    if args.gpu:
        model = model.cuda()
    
    # 創建 PyTorch Lightning 的 Trainer
    # trainer = pl.Trainer(max_epochs=args.num_epochs, gpus=1 if args.gpu else 0)
    trainer = pl.Trainer(max_epochs=args.num_epochs) # , devices=1 if args.gpu else 0
    trainer.fit(model, train_loader)

    # 生成示例文本
    if args.generate_every > 0:
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
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--generate-every", type=int, default=5, help="Generate text every N epochs")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--gpu", action="store_true", help="Use GPU")
    
    args = parser.parse_args()
    main(args)
