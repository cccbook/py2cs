import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
from typing import List, Tuple

class AudioDataset(Dataset):
    def __init__(self, root_dir: str, csv_file: str, transform=None):
        """
        Args:
            root_dir (str): 音訊文件的根目錄
            csv_file (str): 包含音訊文件名和標籤的CSV文件
            transform: 可選的轉換函數
        """
        self.root_dir = root_dir
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        
        # 創建標籤到索引的映射
        self.label2idx = {label: idx for idx, label in enumerate(sorted(set(self.data['label'])))}
        
        # 音訊預處理工具
        self.melspec = T.MelSpectrogram(
            sample_rate=16000,
            n_fft=400,
            win_length=400,
            hop_length=160,
            n_mels=64
        )
        self.amplitude_to_db = T.AmplitudeToDB()
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        audio_path = os.path.join(self.root_dir, self.data.iloc[idx]['file_name'])
        label = self.label2idx[self.data.iloc[idx]['label']]
        
        # 載入音訊
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # 重採樣到 16kHz（如果需要）
        if sample_rate != 16000:
            resampler = T.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
        
        # 轉換為梅爾頻譜圖
        mel_spectrogram = self.melspec(waveform)
        mel_spectrogram = self.amplitude_to_db(mel_spectrogram)
        
        # 應用其他轉換（如果有的話）
        if self.transform:
            mel_spectrogram = self.transform(mel_spectrogram)
        
        return mel_spectrogram, torch.tensor(label, dtype=torch.long)

class SpeechRecognitionModel(nn.Module):
    def __init__(self, num_classes: int, input_channels: int = 1):
        super(SpeechRecognitionModel, self).__init__()
        
        # CNN 特徵提取器
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # BiLSTM 序列處理
        self.lstm = nn.LSTM(
            input_size=128 * 8,  # 根據 CNN 輸出調整
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )
        
        # 分類器
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),  # 512 = hidden_size*2 (雙向)
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        # CNN 特徵提取
        x = self.conv1(x)  # [batch, channels, time, freq]
        x = self.conv2(x)
        x = self.conv3(x)
        
        # 準備 LSTM 輸入
        batch_size, channels, time, freq = x.size()
        x = x.permute(0, 2, 1, 3)  # [batch, time, channels, freq]
        x = x.contiguous().view(batch_size, time, channels * freq)
        
        # BiLSTM 處理
        lstm_out, _ = self.lstm(x)
        
        # 使用最後一個時間步的輸出進行分類
        x = lstm_out[:, -1, :]
        x = self.classifier(x)
        
        return x

def train_model(model: nn.Module,
                train_loader: DataLoader,
                val_loader: DataLoader,
                num_epochs: int,
                device: torch.device) -> Tuple[List[float], List[float]]:
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # 訓練階段
        model.train()
        epoch_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}'
                      f' ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
        
        train_losses.append(epoch_loss / len(train_loader))
        
        # 驗證階段
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        accuracy = 100. * correct / len(val_loader.dataset)
        print(f'Validation set: Average loss: {val_loss:.4f}, '
              f'Accuracy: {correct}/{len(val_loader.dataset)} ({accuracy:.2f}%)')
        
        # 更新學習率
        scheduler.step(val_loss)
    
    return train_losses, val_losses

def main():
    # 設定參數
    batch_size = 32
    num_epochs = 30
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 創建數據集和數據加載器
    # 注意：你需要提供自己的數據集路徑和CSV文件
    train_dataset = AudioDataset(
        root_dir='path/to/audio/files',
        csv_file='path/to/train.csv'
    )
    
    val_dataset = AudioDataset(
        root_dir='path/to/audio/files',
        csv_file='path/to/val.csv'
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # 創建模型
    model = SpeechRecognitionModel(
        num_classes=len(train_dataset.label2idx),
        input_channels=1
    ).to(device)
    
    # 訓練模型
    train_losses, val_losses = train_model(
        model,
        train_loader,
        val_loader,
        num_epochs,
        device
    )
    
    # 保存模型
    torch.save(model.state_dict(), 'speech_recognition_model.pth')

if __name__ == "__main__":
    main()