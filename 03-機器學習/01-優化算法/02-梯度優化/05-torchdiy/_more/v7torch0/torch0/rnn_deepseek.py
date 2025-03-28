import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, nonlinearity='tanh', bias=True, batch_first=True):
        super(RNN, self).__init__()
        assert batch_first, "目前只支持 batch_first=True"
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.nonlinearity = nonlinearity
        self.bias = bias
        
        # 初始化權重和偏置
        self.W_ih = nn.Parameter(torch.randn(hidden_size, input_size))
        self.W_hh = nn.Parameter(torch.randn(hidden_size, hidden_size))
        if bias:
            self.b_ih = nn.Parameter(torch.randn(hidden_size))
            self.b_hh = nn.Parameter(torch.randn(hidden_size))
        else:
            self.register_parameter('b_ih', None)
            self.register_parameter('b_hh', None)

    def forward(self, x, h_0=None):
        """
        手動實現 RNN。
        
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
            # 計算當前時間步的隱藏狀態
            x_t = x[t]  # 當前時間步的輸入，形狀為 (batch_size, input_size)
            h_t = torch.tanh(
                torch.matmul(x_t, self.W_ih.t()) + self.b_ih +
                torch.matmul(h_t, self.W_hh.t()) + self.b_hh
            )
            outputs.append(h_t)
        
        # 將輸出堆疊成一個張量
        output = torch.stack(outputs, dim=0)
        h_n = h_t.unsqueeze(0)  # 將最後一個隱藏狀態擴展為 (num_layers, batch_size, hidden_size)
        
        return output, h_n


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

