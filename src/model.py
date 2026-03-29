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

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, context_length: int = 1024):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        self.n_heads = n_heads
        self.d_head = d_model // n_heads   # dimension per head

        # Project input to Q, K, V in one shot
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        # Pre-allocate causal mask as a buffer so it moves to GPU with the module
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1).bool(),
        )

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
        attn = attn.masked_fill(self.causal_mask[:T, :T], float("-inf"))
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Weighted sum of values, then reassemble heads
        out = attn @ v                              # (B, n_heads, T, d_head)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.

    Applied independently to each token. Expands to d_ff then projects back to d_model.
    GELU activation is used (smoother than ReLU, standard in modern transformers).

    Shape flow:
      input:  (B, T, d_model)
      output: (B, T, d_model)
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    One transformer layer: pre-norm attention + pre-norm FFN, both with residual connections.

    Pre-norm means LayerNorm is applied BEFORE the sub-layer (not after).
    This is the modern convention (used by GPT-2 onward) and trains more stably.

    Shape flow:
      input:  (B, T, d_model)
      output: (B, T, d_model)
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = Attention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))   # attention sub-layer with residual
        x = x + self.ffn(self.ln2(x))    # FFN sub-layer with residual
        return x
