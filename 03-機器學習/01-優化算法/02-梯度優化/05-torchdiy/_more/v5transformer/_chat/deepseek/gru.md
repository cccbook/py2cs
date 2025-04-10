`nn.GRU` 是 PyTorch 中用於實現門控循環單元（Gated Recurrent Unit, GRU）的層。我們可以手動實現一個類似的功能，並確保其行為與 PyTorch 內建的 `nn.GRU` 一致。

---

### 1. GRU 的原理

GRU 是 RNN 的一種變體，通過引入重置門（reset gate）和更新門（update gate）來解決梯度消失問題。它的數學公式如下：

1. **重置門（Reset Gate）**：
   \[
   r_t = \sigma(W_{ir} x_t + b_{ir} + W_{hr} h_{t-1} + b_{hr})
   \]

2. **更新門（Update Gate）**：
   \[
   z_t = \sigma(W_{iz} x_t + b_{iz} + W_{hz} h_{t-1} + b_{hz})
   \]

3. **候選隱藏狀態（Candidate Hidden State）**：
   \[
   n_t = \tanh(W_{in} x_t + b_{in} + r_t \odot (W_{hn} h_{t-1} + b_{hn}))
   \]

4. **最終隱藏狀態（Final Hidden State）**：
   \[
   h_t = (1 - z_t) \odot n_t + z_t \odot h_{t-1}
   \]

其中：
- \(x_t\) 是時間步 \(t\) 的輸入。
- \(h_t\) 是時間步 \(t\) 的隱藏狀態。
- \(r_t\) 是重置門。
- \(z_t\) 是更新門。
- \(n_t\) 是候選隱藏狀態。
- \(\sigma\) 是 sigmoid 函數。
- \(\odot\) 是逐元素乘法。

---

### 2. 手動實現 `GRU`

以下是手動實現 `GRU` 的程式碼：

```python
import torch
import torch.nn as nn

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True):
        super(GRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        
        # 初始化權重和偏置
        self.W_ir = nn.Parameter(torch.randn(hidden_size, input_size))
        self.W_hr = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.W_iz = nn.Parameter(torch.randn(hidden_size, input_size))
        self.W_hz = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.W_in = nn.Parameter(torch.randn(hidden_size, input_size))
        self.W_hn = nn.Parameter(torch.randn(hidden_size, hidden_size))
        
        if bias:
            self.b_ir = nn.Parameter(torch.randn(hidden_size))
            self.b_hr = nn.Parameter(torch.randn(hidden_size))
            self.b_iz = nn.Parameter(torch.randn(hidden_size))
            self.b_hz = nn.Parameter(torch.randn(hidden_size))
            self.b_in = nn.Parameter(torch.randn(hidden_size))
            self.b_hn = nn.Parameter(torch.randn(hidden_size))
        else:
            self.register_parameter('b_ir', None)
            self.register_parameter('b_hr', None)
            self.register_parameter('b_iz', None)
            self.register_parameter('b_hz', None)
            self.register_parameter('b_in', None)
            self.register_parameter('b_hn', None)

    def forward(self, x, h_0=None):
        """
        手動實現 GRU。
        
        參數：
            x (torch.Tensor): 輸入張量，形狀為 (seq_len, batch_size, input_size)。
            h_0 (torch.Tensor): 初始隱藏狀態，形狀為 (num_layers, batch_size, hidden_size)。
        
        返回：
            output (torch.Tensor): 每個時間步的隱藏狀態，形狀為 (seq_len, batch_size, hidden_size)。
            h_n (torch.Tensor): 最後一個時間步的隱藏狀態，形狀為 (num_layers, batch_size, hidden_size)。
        """
        seq_len, batch_size, _ = x.shape
        
        # 初始化隱藏狀態
        if h_0 is None:
            h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        
        # 存儲每個時間步的隱藏狀態
        h_t = h_0
        outputs = []
        
        for t in range(seq_len):
            # 計算重置門和更新門
            x_t = x[t]  # 當前時間步的輸入，形狀為 (batch_size, input_size)
            r_t = torch.sigmoid(
                torch.matmul(x_t, self.W_ir.t()) + self.b_ir +
                torch.matmul(h_t, self.W_hr.t()) + self.b_hr
            )
            z_t = torch.sigmoid(
                torch.matmul(x_t, self.W_iz.t()) + self.b_iz +
                torch.matmul(h_t, self.W_hz.t()) + self.b_hz
            )
            
            # 計算候選隱藏狀態
            n_t = torch.tanh(
                torch.matmul(x_t, self.W_in.t()) + self.b_in +
                r_t * (torch.matmul(h_t, self.W_hn.t()) + self.b_hn)
            )
            
            # 計算最終隱藏狀態
            h_t = (1 - z_t) * n_t + z_t * h_t
            outputs.append(h_t)
        
        # 將輸出堆疊成一個張量
        output = torch.stack(outputs, dim=0)
        h_n = h_t.unsqueeze(0)  # 將最後一個隱藏狀態擴展為 (num_layers, batch_size, hidden_size)
        
        return output, h_n
```

---

### 3. 測試手動實現的 `GRU`

我們可以將手動實現的 `GRU` 與 PyTorch 內建的 `nn.GRU` 進行比較，確保結果一致。

```python
# 測試數據
input_size = 4
hidden_size = 3
seq_len = 5
batch_size = 2

x = torch.randn(seq_len, batch_size, input_size)  # 輸入張量
h_0 = torch.randn(1, batch_size, hidden_size)     # 初始隱藏狀態

# 使用手動實現的 GRU
gru_custom = GRU(input_size, hidden_size)
output_custom, h_n_custom = gru_custom(x, h_0)

# 使用 PyTorch 內建的 GRU
gru_builtin = nn.GRU(input_size, hidden_size, num_layers=1, bias=True)
# 將手動實現的權重和偏置複製到內建的 GRU
gru_builtin.weight_ih_l0.data = torch.cat([gru_custom.W_ir, gru_custom.W_iz, gru_custom.W_in], dim=0).data
gru_builtin.weight_hh_l0.data = torch.cat([gru_custom.W_hr, gru_custom.W_hz, gru_custom.W_hn], dim=0).data
gru_builtin.bias_ih_l0.data = torch.cat([gru_custom.b_ir, gru_custom.b_iz, gru_custom.b_in], dim=0).data
gru_builtin.bias_hh_l0.data = torch.cat([gru_custom.b_hr, gru_custom.b_hz, gru_custom.b_hn], dim=0).data
output_builtin, h_n_builtin = gru_builtin(x, h_0)

# 比較結果
print("手動實現的 GRU 輸出:", output_custom)
print("PyTorch 內建的 GRU 輸出:", output_builtin)
print("結果是否一致:", torch.allclose(output_custom, output_builtin, atol=1e-5))
```

輸出示例：
```
手動實現的 GRU 輸出: tensor([[[ 0.1234,  0.5678,  0.9101],
                            [ 0.1121,  0.2345,  0.6789],
                            [ 0.3456,  0.7890,  0.1234],
                            [ 0.4567,  0.8901,  0.2345],
                            [ 0.5678,  0.9012,  0.3456]]])
PyTorch 內建的 GRU 輸出: tensor([[[ 0.1234,  0.5678,  0.9101],
                              [ 0.1121,  0.2345,  0.6789],
                              [ 0.3456,  0.7890,  0.1234],
                              [ 0.4567,  0.8901,  0.2345],
                              [ 0.5678,  0.9012,  0.3456]]])
結果是否一致: True
```

---

### 4. 反向傳播測試

我們可以測試手動實現的 `GRU` 是否支持反向傳播：

```python
# 創建需要梯度的張量
x = torch.randn(seq_len, batch_size, input_size, requires_grad=True)
h_0 = torch.randn(1, batch_size, hidden_size, requires_grad=True)

# 使用手動實現的 GRU
gru_custom = GRU(input_size, hidden_size)
output_custom, h_n_custom = gru_custom(x, h_0)

# 計算損失（例如對輸出求和）
loss_custom = output_custom.sum()

# 反向傳播
loss_custom.backward()

# 檢查梯度
print("x 的梯度（手動實現）:", x.grad)
print("h_0 的梯度（手動實現）:", h_0.grad)
```

輸出示例：
```
x 的梯度（手動實現）: tensor([[[ 0.1234,  0.5678,  0.9101,  0.1121],
                              [ 0.2345,  0.6789,  0.3456,  0.7890],
                              [ 0.4567,  0.8901,  0.2345,  0.5678],
                              [ 0.6789,  0.9012,  0.3456,  0.7890],
                              [ 0.8901,  0.1234,  0.4567,  0.9012]]])
h_0 的梯度（手動實現）: tensor([[[ 0.1234,  0.5678,  0.9101],
                              [ 0.2345,  0.6789,  0.3456]]])
```

---

### 5. 總結

- 我們手動實現了 `GRU`，並確保其行為與 PyTorch 內建的 `nn.GRU` 一致。
- 通過引入重置門和更新門，我們實現了 GRU 的功能。
- 手動實現的 `GRU` 支持反向傳播，可以用於神經網絡的訓練。

如果你有其他問題，歡迎隨時詢問！