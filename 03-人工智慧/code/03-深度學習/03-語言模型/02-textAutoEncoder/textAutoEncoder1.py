import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchtext.data.utils import get_tokenizer
from collections import Counter
from typing import List, Tuple

class Vocabulary:
    def __init__(self):
        self.word2idx = {'<pad>': 0, '<unk>': 1}
        self.idx2word = ['<pad>', '<unk>']
        self.word_count = Counter()
    
    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = len(self.idx2word)
            self.idx2word.append(word)
        self.word_count[word] += 1
    
    def __len__(self):
        return len(self.idx2word)
    
    def __getitem__(self, word):
        return self.word2idx.get(word, self.word2idx['<unk>'])

class TextDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer, vocab, max_length: int = 50):
        self.texts = texts
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = self.tokenizer(text)
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            tokens = tokens + ['<pad>'] * (self.max_length - len(tokens))
        indices = [self.vocab[token] for token in tokens]
        return torch.tensor(indices, dtype=torch.long)

class TextAutoencoder(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, latent_dim: int, max_length: int):
        super(TextAutoencoder, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.max_length = max_length
        
        # 詞嵌入層
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # 編碼器
        self.encoder_rnn = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        # 潛在空間映射
        self.hidden_to_latent = nn.Linear(hidden_dim * 2, latent_dim)
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim * 2)
        
        # 解碼器
        self.decoder_rnn = nn.LSTM(
            embedding_dim,  # 修改：解碼器只使用詞嵌入作為輸入
            hidden_dim * 2,
            num_layers=1,
            batch_first=True
        )
        
        # 輸出層
        self.output_layer = nn.Linear(hidden_dim * 2, vocab_size)
        
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)
        _, (hidden, _) = self.encoder_rnn(embedded)
        hidden_concat = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        latent = self.hidden_to_latent(hidden_concat)
        return latent
    
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        # 從潛在向量恢復初始隱藏狀態
        hidden = self.latent_to_hidden(latent)
        hidden = hidden.view(1, latent.size(0), -1)  # [1, batch_size, hidden_dim*2]
        
        # 創建解碼器的起始輸入（使用特殊的開始標記的嵌入）
        batch_size = latent.size(0)
        decoder_input = torch.zeros(batch_size, self.max_length, self.embedding_dim).to(latent.device)
        
        # 一次性運行整個序列
        decoder_outputs, _ = self.decoder_rnn(decoder_input, (hidden, torch.zeros_like(hidden)))
        outputs = self.output_layer(decoder_outputs)
        
        return outputs
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(x)
        outputs = self.decode(latent)
        return outputs, latent

def train_model(model: nn.Module,
                train_loader: DataLoader,
                num_epochs: int,
                device: torch.device) -> List[float]:
    
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 0 是 <pad> 的索引
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    losses = []
    model.train()
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            
            optimizer.zero_grad()
            outputs, _ = model(batch)
            loss = criterion(outputs.view(-1, outputs.size(-1)), batch.view(-1))
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
    
    return losses

def create_vocab(texts: List[str], tokenizer) -> Vocabulary:
    vocab = Vocabulary()
    for text in texts:
        tokens = tokenizer(text)
        for token in tokens:
            vocab.add_word(token)
    return vocab

def main():
    # 準備示例數據
    texts = [
        "This is a sample sentence.",
        "Another example text.",
        "Learning to encode and decode text.",
        "The quick brown fox jumps over the lazy dog.",
        "Python programming is fun and interesting.",
        "Natural language processing is fascinating.",
        "Deep learning models can be very powerful.",
        "The weather is nice today.",
        "I love programming in Python.",
        "Machine learning is transforming the world."
    ]
    
    # 初始化分詞器和詞彙表
    tokenizer = get_tokenizer('basic_english')
    vocab = create_vocab(texts, tokenizer)
    
    # 創建數據集和數據加載器
    dataset = TextDataset(texts, tokenizer, vocab)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # 初始化模型
    model = TextAutoencoder(
        vocab_size=len(vocab),
        embedding_dim=128,  # 減小維度
        hidden_dim=256,    # 減小維度
        latent_dim=64,     # 減小維度
        max_length=50
    )
    
    # 訓練模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    losses = train_model(model, dataloader, num_epochs=50, device=device)
    
    # 測試模型
    print("\nTesting reconstruction:")
    model.eval()
    with torch.no_grad():
        for text in texts[:3]:  # 測試前三個句子
            tokens = tokenizer(text)
            if len(tokens) > 50:
                tokens = tokens[:50]
            else:
                tokens = tokens + ['<pad>'] * (50 - len(tokens))
            
            indices = torch.tensor([[vocab[token] for token in tokens]], dtype=torch.long).to(device)
            outputs, _ = model(indices)
            
            # 獲取最可能的詞
            predictions = outputs.argmax(dim=-1)[0]
            reconstructed_text = ' '.join([vocab.idx2word[idx.item()] for idx in predictions 
                                        if vocab.idx2word[idx.item()] not in ['<pad>', '<unk>']])
            print(f"\nOriginal: {text}")
            print(f"Reconstructed: {reconstructed_text}")

if __name__ == "__main__":
    main()