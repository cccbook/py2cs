# 本程式由 ccc 指揮 claude 撰寫
import torch
import torch.nn as nn
import math
from typing import Tuple, Optional, Union

class RNN(nn.Module):
    """
    一個與 PyTorch nn.RNN 相容的自定義 RNN 實現
    
    參數:
        input_size: 輸入特徵維度
        hidden_size: 隱藏狀態維度
        num_layers: RNN 層數
        nonlinearity: 非線性激活函數，可以是 'tanh' 或 'relu'
        bias: 是否使用偏置項
        batch_first: 如果為 True，則輸入和輸出張量的形狀為 (batch, seq, feature)
        dropout: 如果非零，則在除最後一層外的每層輸出上引入一個 dropout 層
        bidirectional: 如果為 True，則變成雙向 RNN
    """
    
    def __init__(self, 
                 input_size: int, 
                 hidden_size: int, 
                 num_layers: int = 1, 
                 nonlinearity: str = 'tanh', 
                 bias: bool = True, 
                 batch_first: bool = False, 
                 dropout: float = 0., 
                 bidirectional: bool = False):
        super(RNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.nonlinearity = nonlinearity
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = float(dropout)
        self.bidirectional = bidirectional
        
        self.num_directions = 2 if bidirectional else 1
        
        # 檢查參數有效性
        if nonlinearity not in ['tanh', 'relu']:
            raise ValueError(f"非線性函數必須是 'tanh' 或 'relu'，得到的是 {nonlinearity}")
            
        if dropout > 0 and num_layers == 1:
            warnings.warn("dropout 參數 > 0，但 num_layers = 1，dropout 將被忽略")
        
        # 創建激活函數
        self.activation = torch.tanh if nonlinearity == 'tanh' else torch.relu
        
        # 定義網絡參數
        self.weight_ih_l = nn.ParameterList()
        self.weight_hh_l = nn.ParameterList()
        self.bias_ih_l = nn.ParameterList() if bias else None
        self.bias_hh_l = nn.ParameterList() if bias else None
        
        # 定義 dropout 層
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else None
        
        # 為每一層初始化參數
        for layer in range(num_layers):
            for direction in range(self.num_directions):
                suffix = '_reverse' if direction == 1 else ''
                
                layer_input_size = input_size if layer == 0 else hidden_size * self.num_directions
                
                # 創建權重和偏置
                weight_ih = nn.Parameter(torch.Tensor(hidden_size, layer_input_size))
                weight_hh = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
                
                self.weight_ih_l.append(weight_ih)
                self.weight_hh_l.append(weight_hh)
                
                if bias:
                    bias_ih = nn.Parameter(torch.Tensor(hidden_size))
                    bias_hh = nn.Parameter(torch.Tensor(hidden_size))
                    self.bias_ih_l.append(bias_ih)
                    self.bias_hh_l.append(bias_hh)
                
        # 初始化參數
        self.reset_parameters()
    
    def reset_parameters(self):
        """初始化所有權重和偏置"""
        stdv = 1.0 / math.sqrt(self.hidden_size)
        
        for weight in self.weight_ih_l:
            nn.init.uniform_(weight, -stdv, stdv)
        
        for weight in self.weight_hh_l:
            nn.init.orthogonal_(weight)
            
        if self.bias:
            for bias in self.bias_ih_l:
                nn.init.uniform_(bias, -stdv, stdv)
            
            for bias in self.bias_hh_l:
                nn.init.zeros_(bias)
                
    def forward(self, 
                input: torch.Tensor, 
                hx: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向傳播
        
        參數:
            input: 輸入序列。如果 batch_first=True 則形狀為 (batch, seq, feature)
                  否則形狀為 (seq, batch, feature)
            hx: 初始隱藏狀態。形狀為 (num_layers * num_directions, batch, hidden_size)
                如果未提供，則初始化為零
                
        返回:
            output: 每個時間步的輸出。如果 batch_first=True 則形狀為 (batch, seq, hidden_size * num_directions)
                    否則形狀為 (seq, batch, hidden_size * num_directions)
            h_n: 最終隱藏狀態。形狀為 (num_layers * num_directions, batch, hidden_size)
        """
        # 處理 batch_first
        if self.batch_first:
            input = input.transpose(0, 1)  # 變成 (seq, batch, feature)
            
        seq_len, batch_size, _ = input.size()
        
        # 如果沒有提供隱藏狀態，則初始化為零
        if hx is None:
            hx = torch.zeros(self.num_layers * self.num_directions, 
                             batch_size, 
                             self.hidden_size, 
                             dtype=input.dtype, 
                             device=input.device)
        
        # 存儲每一層輸出的隱藏狀態
        h_n = torch.zeros_like(hx)
        
        # 存儲每個時間步的輸出
        outputs = []
        
        # 將輸入按層處理
        layer_input = input
        
        for layer in range(self.num_layers):
            # 處理每個方向
            layer_outputs = []
            
            for direction in range(self.num_directions):
                # 獲取該層該方向的初始隱藏狀態
                idx = layer * self.num_directions + direction
                h_0 = hx[idx:idx+1].squeeze(0)
                
                # 獲取該層該方向的參數
                weight_ih = self.weight_ih_l[idx]
                weight_hh = self.weight_hh_l[idx]
                bias_ih = self.bias_ih_l[idx] if self.bias else None
                bias_hh = self.bias_hh_l[idx] if self.bias else None
                
                # 處理時間序列
                direction_outputs = []
                h_t = h_0
                
                # 決定序列處理順序
                seq_indices = range(seq_len)
                if direction == 1:  # 反向處理
                    seq_indices = reversed(seq_indices)
                
                for t in seq_indices:
                    x_t = layer_input[t]
                    
                    # RNN 單元計算
                    gates = torch.mm(x_t, weight_ih.t())
                    if bias_ih is not None:
                        gates += bias_ih
                        
                    gates += torch.mm(h_t, weight_hh.t())
                    if bias_hh is not None:
                        gates += bias_hh
                        
                    h_t = self.activation(gates)
                    direction_outputs.append(h_t)
                
                # 儲存最終隱藏狀態
                h_n[idx] = h_t
                
                # 如果是反向，需要反轉輸出序列
                if direction == 1:
                    direction_outputs = list(reversed(direction_outputs))
                
                # 將輸出堆疊成張量
                direction_output = torch.stack(direction_outputs, dim=0)
                layer_outputs.append(direction_output)
            
            # 將正向和反向輸出連接起來
            if self.num_directions == 2:
                layer_output = torch.cat(layer_outputs, dim=2)
            else:
                layer_output = layer_outputs[0]
            
            # 更新下一層的輸入
            layer_input = layer_output
            
            # 應用 dropout (除了最後一層)
            if self.dropout_layer is not None and layer < self.num_layers - 1:
                layer_input = self.dropout_layer(layer_input)
        
        # 處理最終輸出
        output = layer_input
        
        # 如果 batch_first=True，則需要轉置輸出
        if self.batch_first:
            output = output.transpose(0, 1)
            
        return output, h_n
    
    def extra_repr(self) -> str:
        """返回實例的額外表示信息"""
        s = '{input_size}, {hidden_size}'
        if self.num_layers != 1:
            s += ', num_layers={num_layers}'
        if self.nonlinearity != 'tanh':
            s += ', nonlinearity={nonlinearity}'
        if not self.bias:
            s += ', bias={bias}'
        if self.batch_first:
            s += ', batch_first={batch_first}'
        if self.dropout > 0:
            s += ', dropout={dropout}'
        if self.bidirectional:
            s += ', bidirectional={bidirectional}'
        return s.format(**self.__dict__)

