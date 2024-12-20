import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class CharRNN(nn.Module):
    """簡單的字符級 RNN"""
    def __init__(self, vocab_size: int, hidden_size: int = 256):
        super(CharRNN, self).__init__()
        
        self.hidden_size = hidden_size
        
        # 字符嵌入層
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        
        # RNN 層
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
        
        # 輸出層
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x: torch.Tensor, h: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # x shape: (batch_size, sequence_length)
        x = self.embedding(x)  # shape: (batch_size, sequence_length, hidden_size)
        
        # RNN 前向傳播
        out, h = self.rnn(x, h)  # out: (batch_size, sequence_length, hidden_size)
        
        # 應用輸出層
        logits = self.fc(out)  # shape: (batch_size, sequence_length, vocab_size)
        
        return logits, h

def create_sequences(text: str, seq_length: int) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    """創建訓練序列和字符映射"""
    chars = sorted(list(set(text)))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}

    sequences = [text[i:i + seq_length + 1] for i in range(len(text) - seq_length)]
    return sequences, char_to_idx, idx_to_char

def prepare_batch(sequences: List[str], char_to_idx: Dict[str, int], batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """準備一個批次的數據"""
    batch_sequences = np.random.choice(sequences, batch_size)
    X = torch.tensor([[char_to_idx[c] for c in seq[:-1]] for seq in batch_sequences], dtype=torch.long)
    y = torch.tensor([[char_to_idx[c] for c in seq[1:]] for seq in batch_sequences], dtype=torch.long)
    return X, y

def loss_fn(model: nn.Module, X: torch.Tensor, y: torch.Tensor, criterion: nn.CrossEntropyLoss) -> torch.Tensor:
    """計算損失"""
    logits, _ = model(X)
    logits = logits.reshape(-1, logits.size(-1))
    targets = y.reshape(-1)
    return criterion(logits, targets)

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
    generated = seed_text
    h = None

    x = torch.tensor([[char_to_idx[c] for c in seed_text]], dtype=torch.long)

    for _ in range(length):
        logits, h = model(x, h)
        logits = logits[:, -1, :] / temperature
        probs = torch.softmax(logits, dim=-1).squeeze().detach().cpu().numpy()
        next_char_idx = np.random.choice(len(probs), p=probs)
        generated += idx_to_char[next_char_idx]
        x = torch.tensor([[next_char_idx]], dtype=torch.long)

    return generated

def main(args):
    with open(args.input_file, 'r', encoding='utf-8') as f:
        text = f.read()

    sequences, char_to_idx, idx_to_char = create_sequences(text, args.sequence_length)
    vocab_size = len(char_to_idx)
    print(f"Vocabulary size: {vocab_size}")

    model = CharRNN(vocab_size, args.hidden_size)
    if args.gpu and torch.cuda.is_available():
        model = model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0

        for _ in range(args.batches_per_epoch):
            X, y = prepare_batch(sequences, char_to_idx, args.batch_size)
            if args.gpu and torch.cuda.is_available():
                X, y = X.cuda(), y.cuda()

            optimizer.zero_grad()
            loss = loss_fn(model, X, y, criterion)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / args.batches_per_epoch
        print(f"Epoch {epoch + 1}: avg_loss = {avg_loss:.4f}")

        if (epoch + 1) % args.generate_every == 0:
            seed = text[:args.sequence_length]
            generated = generate_text(model, seed, char_to_idx, idx_to_char, length=100, temperature=args.temperature)
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
