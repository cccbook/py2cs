import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        # 前饋神經網絡
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # 層歸一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """
        手動實現 TransformerDecoderLayer。
        
        參數：
            tgt (torch.Tensor): 目標序列，形狀為 (seq_len, batch_size, d_model)。
            memory (torch.Tensor): 編碼器的輸出，形狀為 (seq_len, batch_size, d_model)。
            tgt_mask (torch.Tensor): 目標序列的注意力 mask，形狀為 (seq_len, seq_len)。
            memory_mask (torch.Tensor): 編碼器輸出的注意力 mask，形狀為 (seq_len, seq_len)。
            tgt_key_padding_mask (torch.Tensor): 目標序列的填充 mask，形狀為 (batch_size, seq_len)。
            memory_key_padding_mask (torch.Tensor): 編碼器輸出的填充 mask，形狀為 (batch_size, seq_len)。
        
        返回：
            torch.Tensor: 解碼器層的輸出，形狀為 (seq_len, batch_size, d_model)。
        """
        # 自注意力機制
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        # 交叉注意力機制
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        # 前饋神經網絡
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        
        return tgt

class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([decoder_layer for _ in range(num_layers)])

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """
        手動實現 TransformerDecoder。
        
        參數：
            tgt (torch.Tensor): 目標序列，形狀為 (seq_len, batch_size, d_model)。
            memory (torch.Tensor): 編碼器的輸出，形狀為 (seq_len, batch_size, d_model)。
            tgt_mask (torch.Tensor): 目標序列的注意力 mask，形狀為 (seq_len, seq_len)。
            memory_mask (torch.Tensor): 編碼器輸出的注意力 mask，形狀為 (seq_len, seq_len)。
            tgt_key_padding_mask (torch.Tensor): 目標序列的填充 mask，形狀為 (batch_size, seq_len)。
            memory_key_padding_mask (torch.Tensor): 編碼器輸出的填充 mask，形狀為 (batch_size, seq_len)。
        
        返回：
            torch.Tensor: 解碼器的輸出，形狀為 (seq_len, batch_size, d_model)。
        """
        output = tgt
        for layer in self.layers:
            output = layer(output, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask)
        return output
