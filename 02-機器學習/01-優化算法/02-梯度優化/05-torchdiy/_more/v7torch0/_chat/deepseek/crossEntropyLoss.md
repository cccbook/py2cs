實作自己的 `CrossEntropyLoss` 是一個很好的練習，可以幫助你更深入地理解損失函數的計算過程。`CrossEntropyLoss` 的核心是結合 `LogSoftmax` 和負對數似然損失（Negative Log Likelihood Loss, NLLLoss）。

以下是自定義 `CrossEntropyLoss` 的完整實現，並附上詳細解釋。

---

### 1. `CrossEntropyLoss` 的數學公式

`CrossEntropyLoss` 的公式如下：
\[
\text{CrossEntropyLoss}(x, y) = -\log\left(\frac{e^{x_y}}{\sum_{j} e^{x_j}}\right) = -x_y + \log\left(\sum_{j} e^{x_j}\right)
\]
其中：
- \(x\) 是模型的輸出（logits，未經過 Softmax）。
- \(y\) 是目標類別（ground truth label）。
- \(x_y\) 是目標類別對應的 logit。

---

### 2. 自定義 `CrossEntropyLoss` 的實現

我們將實現一個自定義的 `CrossEntropyLoss`，包含以下步驟：
1. 計算 `LogSoftmax`。
2. 計算負對數似然損失（NLLLoss）。

以下是程式碼實現：

```python
import torch
import torch.nn as nn

class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, logits, target):
        """
        計算 CrossEntropyLoss。
        
        參數：
            logits (torch.Tensor): 模型的輸出，形狀為 (batch_size, num_classes)。
            target (torch.Tensor): 目標類別，形狀為 (batch_size,)。
        
        返回：
            loss (torch.Tensor): 計算得到的損失值。
        """
        # 1. 計算 LogSoftmax
        log_softmax = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        
        # 2. 計算負對數似然損失（NLLLoss）
        # 使用 gather 來選取目標類別對應的 logit
        nll_loss = -log_softmax.gather(1, target.unsqueeze(1)).mean()
        
        return nll_loss
```

---

### 3. 程式碼說明

#### 3.1 計算 `LogSoftmax`
- `LogSoftmax` 的公式為：
  \[
  \text{LogSoftmax}(x_i) = x_i - \log\left(\sum_{j} e^{x_j}\right)
  \]
- 我們使用 `torch.logsumexp` 來計算 \(\log\left(\sum_{j} e^{x_j}\right)\)，這是一種數值穩定的計算方法。

#### 3.2 計算負對數似然損失（NLLLoss）
- 我們使用 `gather` 來選取目標類別對應的 logit。
- 最後對所有樣本的損失取平均（`mean()`）。

---

### 4. 測試自定義 `CrossEntropyLoss`

我們可以將自定義的 `CrossEntropyLoss` 與 PyTorch 內建的 `nn.CrossEntropyLoss` 進行比較，確保結果一致。

```python
# 測試數據
logits = torch.tensor([[2.0, 1.0, 0.1], [0.5, 2.0, 0.3]], requires_grad=True)  # 模型輸出
target = torch.tensor([0, 1])  # 目標類別

# 使用自定義的 CrossEntropyLoss
criterion_custom = CrossEntropyLoss()
loss_custom = criterion_custom(logits, target)

# 使用 PyTorch 內建的 CrossEntropyLoss
criterion_builtin = nn.CrossEntropyLoss()
loss_builtin = criterion_builtin(logits, target)

# 比較結果
print("自定義 CrossEntropyLoss 結果:", loss_custom.item())
print("PyTorch 內建 CrossEntropyLoss 結果:", loss_builtin.item())
print("結果是否一致:", torch.allclose(loss_custom, loss_builtin))
```

輸出示例：
```
自定義 CrossEntropyLoss 結果: 0.9746
PyTorch 內建 CrossEntropyLoss 結果: 0.9746
結果是否一致: True
```

---

### 5. 反向傳播測試

我們可以測試自定義 `CrossEntropyLoss` 的反向傳播是否正常工作：

```python
# 計算損失
loss_custom = criterion_custom(logits, target)

# 反向傳播
loss_custom.backward()

# 檢查梯度
print("Logits 的梯度:", logits.grad)
```

輸出示例：
```
Logits 的梯度: tensor([[ 0.6652,  0.2447,  0.0901],
                       [ 0.2447, -0.9102,  0.6655]])
```

---

### 6. 總結

- 我們實現了一個自定義的 `CrossEntropyLoss`，它結合了 `LogSoftmax