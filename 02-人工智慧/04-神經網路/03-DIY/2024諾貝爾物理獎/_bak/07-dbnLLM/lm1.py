import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data import Field, Dataset, Example, BucketIterator
from sklearn.model_selection import train_test_split
import random

# 設置隨機種子以便重現
torch.manual_seed(42)
random.seed(42)

# 數據集準備
def prepare_data(text, k):
    words = text.split()
    inputs, targets = [], []

    for i in range(len(words) - k):
        inputs.append(words[i:i + k])
        targets.append(words[i + k])
    
    return inputs, targets

# 測試文本數據
text_data = "這是範例文本，用於展示如何建立語言模型。我們將使用前幾個詞預測下一個詞。"
k = 3  # 前幾個詞的數量
inputs, targets = prepare_data(text_data, k)

# 建立詞彙表
word_list = list(set([word for word in inputs + targets]))
word_to_ix = {word: i for i, word in enumerate(word_list)}
ix_to_word = {i: word for i, word in enumerate(word_list)}

# 數據轉換
X = [[word_to_ix[word] for word in seq] for seq in inputs]
y = [word_to_ix[target] for target in targets]

# 切分數據集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 轉換為 Tensor
X_train_tensor = torch.LongTensor(X_train)
y_train_tensor = torch.LongTensor(y_train)
X_test_tensor = torch.LongTensor(X_test)
y_test_tensor = torch.LongTensor(y_test)

# 定義語言模型
class LanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(LanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)  # 嵌入層
        x, _ = self.rnn(x)     # RNN 層
        x = self.fc(x[:, -1, :])  # 取最後一個時間步的輸出
        return x

# 設置模型參數
vocab_size = len(word_list)
embed_size = 10
hidden_size = 10

model = LanguageModel(vocab_size, embed_size, hidden_size)

# 設置損失函數和優化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 訓練模型
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    
    # 前向傳播
    outputs = model(X_train_tensor)
    
    # 計算損失
    loss = criterion(outputs, y_train_tensor)
    
    # 反向傳播
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 測試模型
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    _, predicted = torch.max(test_outputs, 1)
    test_loss = criterion(test_outputs, y_test_tensor)
    print(f'Test Loss: {test_loss.item():.4f}')
    
    # 測試預測
    for i in range(5):
        print(f'Input: {inputs[i]}, Predicted: {ix_to_word[predicted[i].item()]}, Target: {targets[i]}')
