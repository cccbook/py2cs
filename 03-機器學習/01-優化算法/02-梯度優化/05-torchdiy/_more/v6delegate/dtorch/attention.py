import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim 必須能被 num_heads 整除"
        
        # 線性變換的權重和偏置
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.W_o = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        """
        手動實現 MultiheadAttention。
        
        參數：
            query (torch.Tensor): 查詢張量，形狀為 (seq_len_q, batch_size, embed_dim)。
            key (torch.Tensor): 鍵張量，形狀為 (seq_len_k, batch_size, embed_dim)。
            value (torch.Tensor): 值張量，形狀為 (seq_len_k, batch_size, embed_dim)。
            key_padding_mask (torch.Tensor): 鍵的填充 mask，形狀為 (batch_size, seq_len_k)。
            attn_mask (torch.Tensor): 注意力 mask，形狀為 (seq_len_q, seq_len_k)。
        
        返回：
            output (torch.Tensor): 注意力輸出，形狀為 (seq_len_q, batch_size, embed_dim)。
            attn_weights (torch.Tensor): 注意力權重，形狀為 (batch_size, num_heads, seq_len_q, seq_len_k)。
        """
        seq_len_q, batch_size, embed_dim = query.shape
        seq_len_k = key.size(0)
        assert seq_len_k == seq_len_q, f"seq_len_q 和 seq_len_k 必須相等, 但 seq_len_q={seq_len_q}, seq_len_k={seq_len_k}"
        
        # 線性變換
        Q = self.W_q(query)  # (seq_len_q, batch_size, embed_dim)
        K = self.W_k(key)    # (seq_len_k, batch_size, embed_dim)
        V = self.W_v(value)  # (seq_len_k, batch_size, embed_dim)
        
        # 將 Q, K, V 分成多個頭
        Q = Q.view(seq_len_q, batch_size, self.num_heads, self.head_dim).transpose(0, 1)  # (batch_size, seq_len_q, num_heads, head_dim)
        K = K.view(seq_len_k, batch_size, self.num_heads, self.head_dim).transpose(0, 1)  # (batch_size, seq_len_k, num_heads, head_dim)
        V = V.view(seq_len_k, batch_size, self.num_heads, self.head_dim).transpose(0, 1)  # (batch_size, seq_len_k, num_heads, head_dim)
        
        # 計算縮放點積注意力
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (batch_size, num_heads, seq_len_q, seq_len_k)
        
        # 應用注意力 mask
        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(attn_mask == 0, float('-inf'))
        
        # 應用鍵的填充 mask
        if key_padding_mask is not None:
            attn_scores = attn_scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        
        # 計算注意力權重
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 計算加權值
        output = torch.matmul(attn_weights, V)  # (batch_size, num_heads, seq_len_q, head_dim)
        
        # 將多個頭的輸出拼接起來
        output = output.transpose(0, 1).contiguous().view(seq_len_q, batch_size, embed_dim)  # (seq_len_q, batch_size, embed_dim)
        
        # 線性變換
        output = self.W_o(output)
        
        return output, attn_weights