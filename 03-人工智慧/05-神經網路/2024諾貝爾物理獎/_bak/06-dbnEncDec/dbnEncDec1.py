import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

# 定義 Encoder-Decoder 結構
class EncoderDecoder(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes):
        super(EncoderDecoder, self).__init__()
        self.encoder = DBN(input_size, hidden_sizes)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_sizes[-1], output_size),
            nn.Sigmoid()  # 假設輸出範圍在 0 到 1 之間
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# 創建數據集（這裡可以自定義輸入和輸出的維度）
n=10 # n = 2  # 輸入維度
m=5 # m = 1  # 輸出維度
X, y = make_moons(n_samples=1000, noise=0.1)

# 調整 y 的形狀以匹配輸出維度
y = y.reshape(-1, m)

# 分割數據集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 數據標準化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 轉換為 Tensor
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test)

# 設置模型參數
hidden_sizes = [5, 3]  # 可以根據需求調整隱藏層的大小
model = EncoderDecoder(input_size=n, output_size=m, hidden_sizes=hidden_sizes)

# 設置損失函數和優化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 訓練模型
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 測試模型
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    test_loss = criterion(test_outputs, y_test_tensor)
    print(f'Test Loss: {test_loss.item():.4f}')
