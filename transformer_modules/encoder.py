import torch
import torch.nn as nn
from transformer_modules.mha import MultiHeadAttention
from transformer_modules.ff import FeedForward


class EncoderBlock(nn.Module):
    def __init__(self, embeding_dim, num_heads, drop_out):
        super().__init__()
        head_size = embeding_dim// num_heads
        self.sa = MultiHeadAttention(num_heads, head_size, embeding_dim, drop_out)
        self.ffwd = FeedForward(embeding_dim, drop_out)
        self.ln1 = nn.LayerNorm(embeding_dim)
        self.ln2 = nn.LayerNorm(embeding_dim)

    def forward(self, x, padding_mask):
        x = x + self.sa(self.ln1(x), padding_mask)
        x = x + self.ffwd(self.ln2(x))
        return x