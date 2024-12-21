import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# 設定隨機種子確保結果可重現
torch.manual_seed(42)

# 設定設備
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 定義自編碼器模型
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        
        # 編碼器
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 32)
        )
        
        # 解碼器
        self.decoder = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 28*28),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = x.view(-1, 28*28)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded.view(-1, 1, 28, 28)

# 資料預處理
transform = transforms.Compose([
    transforms.ToTensor()
])

# 載入 MNIST 資料集
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# 初始化模型、損失函數和優化器
model = Autoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 訓練函數
def train(model, train_loader, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            
            # 前向傳播
            output = model(data)
            loss = criterion(output, data)
            
            # 反向傳播和優化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/num_epochs], Average Loss: {avg_loss:.4f}')

# 視覺化結果函數
def visualize_reconstruction(model, data_loader):
    model.eval()
    with torch.no_grad():
        # 獲取一批數據
        data, _ = next(iter(data_loader))
        data = data.to(device)
        
        # 重建圖像
        reconstruction = model(data)
        
        # 顯示原始圖像和重建圖像
        n = 10
        plt.figure(figsize=(20, 4))
        for i in range(n):
            # 原始圖像
            ax = plt.subplot(2, n, i + 1)
            plt.imshow(data[i].cpu().squeeze(), cmap='gray')
            plt.axis('off')
            
            # 重建圖像
            ax = plt.subplot(2, n, i + 1 + n)
            plt.imshow(reconstruction[i].cpu().squeeze(), cmap='gray')
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()

# 主程式
if __name__ == "__main__":
    num_epochs = 10
    
    # 訓練模型
    train(model, train_loader, num_epochs)
    
    # 視覺化結果
    visualize_reconstruction(model, train_loader)