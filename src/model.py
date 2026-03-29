import math
import torch
import torch.nn as nn


class Attention(nn.Module):
    """
    Multi-head causal self-attention.

    Each token attends to all previous tokens (and itself) but not future ones.
    This is enforced by an upper-triangular mask filled with -inf before softmax.

    Shape flow:
      input:  (B, T, d_model)
      output: (B, T, d_model)
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        self.n_heads = n_heads
        self.d_head = d_model // n_heads   # dimension per head

        # Project input to Q, K, V in one shot
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape  # batch, sequence length, d_model

        # Compute Q, K, V and split
        qkv = self.qkv(x)                           # (B, T, 3*C)
        q, k, v = qkv.split(C, dim=2)              # each: (B, T, C)

        # Reshape to (B, n_heads, T, d_head) for per-head attention
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # Scaled dot-product attention: softmax(QK^T / sqrt(d_head)) * V
        scale = math.sqrt(self.d_head)
        attn = (q @ k.transpose(-2, -1)) / scale   # (B, n_heads, T, T)

        # Causal mask: positions can only attend to positions <= themselves
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float("-inf"))
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Weighted sum of values, then reassemble heads
        out = attn @ v                              # (B, n_heads, T, d_head)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)
