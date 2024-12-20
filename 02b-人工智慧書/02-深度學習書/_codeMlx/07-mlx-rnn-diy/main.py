import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

class CharRNN(nn.Module):
    """簡單的字符級 RNN"""
    def __init__(self, vocab_size: int, hidden_size: int = 256):
        super().__init__()
        
        self.hidden_size = hidden_size
        
        # 字符嵌入層
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        
        # RNN 層
        self.Wx = mx.random.normal((hidden_size, hidden_size), scale=0.01)
        self.Wh = mx.random.normal((hidden_size, hidden_size), scale=0.01)
        self.b = mx.zeros((hidden_size,))
        
        # 輸出層
        self.fc = nn.Linear(hidden_size, vocab_size)
        
    def rnn_step(self, x: mx.array, h: mx.array) -> mx.array:
        """單個 RNN 步驟"""
        return mx.tanh(x @ self.Wx + h @ self.Wh + self.b)
    
    def __call__(self, x: mx.array, h: mx.array = None) -> Tuple[mx.array, mx.array]:
        # x shape: (batch_size, sequence_length)
        batch_size, seq_length = x.shape
        
        # 獲取嵌入
        x = self.embedding(x)  # shape: (batch_size, sequence_length, hidden_size)
        
        # 初始化隱藏狀態
        if h is None:
            h = mx.zeros((batch_size, self.hidden_size))
        
        # RNN 前向傳播
        outputs = []
        for t in range(seq_length):
            h = self.rnn_step(x[:, t], h)
            outputs.append(h)
        
        # 堆疊輸出
        output = mx.stack(outputs, axis=1)  # shape: (batch_size, sequence_length, hidden_size)
        
        # 應用輸出層
        logits = self.fc(output)  # shape: (batch_size, sequence_length, vocab_size)
        
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

def prepare_batch(sequences: List[str], char_to_idx: Dict[str, int], batch_size: int) -> Tuple[mx.array, mx.array]:
    """準備一個批次的數據"""
    # 隨機選擇序列
    batch_sequences = np.random.choice(sequences, batch_size)
    
    # 轉換為索引
    X = np.array([[char_to_idx[c] for c in seq[:-1]] for seq in batch_sequences])
    y = np.array([[char_to_idx[c] for c in seq[1:]] for seq in batch_sequences])
    
    return mx.array(X), mx.array(y)

def loss_fn(model: nn.Module, X: mx.array, y: mx.array) -> mx.array:
    """計算損失"""
    logits, _ = model(X)
    # 重塑 logits 和目標以匹配交叉熵的預期形狀
    logits_2d = logits.reshape(-1, logits.shape[-1])  # (batch_size * seq_length, vocab_size)
    targets_1d = y.reshape(-1)  # (batch_size * seq_length,)
    return mx.mean(nn.losses.cross_entropy(logits_2d, targets_1d))

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
    x = mx.array([[char_to_idx[c] for c in seed_text]])
    
    # 生成字符
    for _ in range(length):
        # 獲取預測
        logits, h = model(x, h)
        logits = logits[:, -1, :] / temperature  # 只使用最後一個時間步
        
        # 計算概率
        probs = mx.softmax(logits, axis=-1)
        probs = probs.tolist()[0]
        
        # 確保概率和為 1
        probs = [p / sum(probs) for p in probs]
        
        # 採樣下一個字符
        next_char_idx = np.random.choice(len(idx_to_char), p=probs)
        
        # 添加到生成的文本
        generated += idx_to_char[next_char_idx]
        
        # 更新輸入
        x = mx.array([[next_char_idx]])
    
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
    
    # 設置優化器
    optimizer = optim.Adam(learning_rate=args.learning_rate)
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    
    # 訓練循環
    for epoch in range(args.num_epochs):
        tic = time.perf_counter()
        total_loss = 0
        num_batches = args.batches_per_epoch
        
        for _ in range(num_batches):
            # 準備批次
            X, y = prepare_batch(sequences, char_to_idx, args.batch_size)
            
            # 計算損失和梯度
            loss, grads = loss_and_grad_fn(model, X, y)
            
            # 更新參數
            optimizer.update(model, grads)
            
            total_loss += float(loss)
        
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
    
    if args.gpu:
        mx.set_default_device(mx.gpu)
    
    main(args)
