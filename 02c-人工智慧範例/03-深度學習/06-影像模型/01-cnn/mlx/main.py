# Copyright 2023 Apple Inc.

import argparse
import time
from functools import partial

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

import mnist

class CNN(nn.Module):
    """A simple CNN for MNIST."""
    def __init__(self):
        super().__init__()
        # 第一個卷積塊
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm(32)
        
        # 第二個卷積塊
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm(64)
        
        # 池化層
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 全連接層
        self.fc1 = nn.Linear(3136, 128)  # 64 * 7 * 7 = 3136 (最後的特徵圖大小)
        self.fc2 = nn.Linear(128, 10)
        
        # Dropout
        self.dropout = nn.Dropout(p=0.5)
        
    def __call__(self, x, training=True):
        # 重塑輸入為 [batch_size, height, width, channels]
        batch_size = x.shape[0]
        x = x.reshape(batch_size, 28, 28, 1)  # NHWC 格式
        
        # 第一個卷積塊
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.relu(x)
        x = self.pool(x)  # 14x14
        
        # 第二個卷積塊
        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.relu(x)
        x = self.pool(x)  # 7x7
        
        # 展平
        x = x.reshape(batch_size, -1)
        
        # 全連接層
        x = nn.relu(self.fc1(x))
        if training:
            x = self.dropout(x)
        x = self.fc2(x)
        
        return x

def loss_fn(model, X, y):
    pred = model(X)
    return nn.losses.cross_entropy(pred, y, reduction="mean")

def batch_iterate(batch_size, X, y):
    perm = mx.array(np.random.permutation(y.size))
    for s in range(0, y.size, batch_size):
        ids = perm[s : s + batch_size]
        yield X[ids], y[ids]

def main(args):
    # 設定超參數
    seed = 0
    batch_size = 128
    num_epochs = 20
    learning_rate = 0.0005

    np.random.seed(seed)

    # 載入數據
    train_images, train_labels, test_images, test_labels = map(
        mx.array, getattr(mnist, args.dataset)()
    )

    # 正規化圖像數據到 [0, 1]
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # 創建模型
    model = CNN()

    # 設定優化器
    optimizer = optim.Adam(learning_rate=learning_rate)
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    # 定義訓練步驟
    def step(model, X, y):
        loss, grads = loss_and_grad_fn(model, X, y)
        optimizer.update(model, grads)
        return loss

    # 定義評估函數
    def eval_fn(model, X, y):
        # 在評估時關閉 dropout
        pred = model(X, training=False)
        return mx.mean(mx.argmax(pred, axis=1) == y)

    # 訓練循環
    best_accuracy = 0.0
    for e in range(num_epochs):
        tic = time.perf_counter()
        total_loss = 0.0
        num_batches = 0
        
        # 訓練一個 epoch
        for X, y in batch_iterate(batch_size, train_images, train_labels):
            loss = step(model, X, y)
            total_loss += float(loss)
            num_batches += 1
        
        # 計算平均損失
        avg_loss = total_loss / num_batches
        
        # 評估測試集
        accuracy = eval_fn(model, test_images, test_labels)
        toc = time.perf_counter()
        
        # 更新最佳準確率
        best_accuracy = max(best_accuracy, accuracy.item())
        
        print(
            f"Epoch {e+1}: Test accuracy {accuracy.item():.3f}, "
            f"Best accuracy {best_accuracy:.3f}, "
            f"Avg loss {avg_loss:.3f}, "
            f"Time {toc - tic:.3f} (s)"
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train a CNN on MNIST with MLX.")
    parser.add_argument("--gpu", action="store_true", help="Use the Metal back-end.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        choices=["mnist", "fashion_mnist"],
        help="The dataset to use.",
    )
    args = parser.parse_args()
    main(args)
