import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import random

# 設置隨機種子以便重現
torch.manual_seed(42)
random.seed(42)

# 數據集準備
def prepare_data(text, k):
    # words = text.split()
    words = [x for x in text]
    print('words=', words)
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
word_array = [word for word in text_data]
print('word_array=', word_array)
word_set = set([word for word in text_data])
print('word_set=', word_set)
word_list = list(set([word for word in text_data]))
word_to_ix = {word: i for i, word in enumerate(word_list)}
ix_to_word = {i: word for i, word in enumerate(word_list)}

# 數據轉換
X = [[word_to_ix[word] for word in seq] for seq in inputs]
y = [word_to_ix[target] for target in targets]
print('X=', X)
print('y=', y)
# 切分數據集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 轉換為 Tensor
X_train_tensor = torch.LongTensor(X_train)
y_train_tensor = torch.LongTensor(y_train)
X_test_tensor = torch.LongTensor(X_test)
y_test_tensor = torch.LongTensor(y_test)

# 定義 DBN 結構
class DBN(nn.Module):
    def __init__(self, input_size, hidden_sizes):
        super(DBN, self).__init__()
        self.layers = nn.ModuleList()
        previous_size = input_size
        
        for hidden_size in hidden_sizes:
            self.layers.append(nn.Sequential(
                nn.Linear(previous_size, hidden_size),
                nn.ReLU()
            ))
            previous_size = hidden_size

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# 定義 MLP 解碼器
class MLPDecoder(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes):
        super(MLPDecoder, self).__init__()
        self.layers = nn.ModuleList()
        previous_size = input_size
        
        for hidden_size in hidden_sizes:
            self.layers.append(nn.Sequential(
                nn.Linear(previous_size, hidden_size),
                nn.ReLU()
            ))
            previous_size = hidden_size
            
        self.output_layer = nn.Linear(previous_size, output_size)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.output_layer(x)

# 定義 Encoder-Decoder 結構
class EncoderDecoder(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes_encoder, hidden_sizes_decoder):
        super(EncoderDecoder, self).__init__()
        self.encoder = DBN(input_size, hidden_sizes_encoder)
        self.decoder = MLPDecoder(hidden_sizes_encoder[-1], output_size, hidden_sizes_decoder)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# 設置模型參數
input_size = len(word_list)  # 輸入維度（詞彙表大小）
output_size = len(word_list)  # 輸出維度（詞彙表大小）
hidden_sizes_encoder = [10, 5]  # Encoder 的隱藏層大小
hidden_sizes_decoder = [10, 5]   # Decoder 的隱藏層大小

# 構建模型
model = EncoderDecoder(input_size, output_size, hidden_sizes_encoder, hidden_sizes_decoder)

# 設置損失函數和優化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 訓練模型
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    
    # 將輸入轉換為 one-hot 編碼
    X_onehot = torch.zeros(X_train_tensor.size(0), input_size)
    print('X_train_tensor=', X_train_tensor)
    print(torch.arange(X_train_tensor.size(0)))
    print(X_train_tensor.flatten())
    X_onehot[torch.arange(X_train_tensor.size(0)), X_train_tensor.flatten()] = 1
    # 前向傳播
    outputs = model(X_onehot)
    
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
    # 將測試輸入轉換為 one-hot 編碼
    X_test_onehot = torch.zeros(X_test_tensor.size(0), input_size)
    X_test_onehot[torch.arange(X_test_tensor.size(0)), X_test_tensor] = 1
    
    test_outputs = model(X_test_onehot)
    test_loss = criterion(test_outputs, y_test_tensor)
    print(f'Test Loss: {test_loss.item():.4f}')
    
    # 測試預測
    _, predicted = torch.max(test_outputs, 1)
    for i in range(5):
        print(f'Input: {inputs[i]}, Predicted: {ix_to_word[predicted[i].item()]}, Target: {targets[i]}')
