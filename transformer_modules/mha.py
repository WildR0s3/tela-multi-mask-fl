import torch
import torch.nn as nn
from torch.nn import functional as F


class Head(nn.Module):
    def __init__(self, head_size, embeding_dim, drop_out):
        super().__init__()
        self.key = nn.Linear(embeding_dim, head_size, bias=False)
        self.query = nn.Linear(embeding_dim, head_size, bias=False)
        self.value = nn.Linear(embeding_dim, head_size, bias=False)
        self.dropout = nn.Dropout(drop_out)

    def forward(self, x, padding_mask=None):
        B, T, C = x.shape
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
        
        if padding_mask is not None:
        # We only care about masking the columns (the keys being looked at)
        # so that no token attends to a PAD/TERM token.
            padding_mask = (padding_mask == 0).unsqueeze(1) # (B, 1, T)
        else:
            padding_mask = 0
        
        wei = wei.masked_fill(padding_mask, float('-inf'))

        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        v = self.value(x) # (B, T, head_size)
        out = wei @ v # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, embeding_dim, drop_out):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, embeding_dim, drop_out) for _ in range(num_heads)])
        self.projection = nn.Linear(num_heads * head_size, embeding_dim)
        self.dropout = nn.Dropout(drop_out)

    def forward(self, x , padding_mask):
        out = torch.cat([h(x, padding_mask) for h in self.heads], dim=-1)
        out = self.projection(out)
        out = self.dropout(out)
        return out