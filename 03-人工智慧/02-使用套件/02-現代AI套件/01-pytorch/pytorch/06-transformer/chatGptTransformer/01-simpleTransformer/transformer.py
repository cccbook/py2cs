import torch
from torch import nn

class Transformer(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads, num_layers):
        super(Transformer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=hidden_size, nhead=num_heads), num_layers=num_layers)
        self.decoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=hidden_size, nhead=num_heads), num_layers=num_layers)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        output = self.encoder(src, src_mask)
        output = self.decoder(tgt, output, tgt_mask, src_mask)
        return output
