import torch
import pytest
from src.model import Attention

def test_attention_output_shape():
    attn = Attention(d_model=64, n_heads=4)
    x = torch.randn(2, 8, 64)   # (batch=2, seq=8, d_model=64)
    out = attn(x)
    assert out.shape == (2, 8, 64)

def test_attention_causal_mask():
    """
    Changing a future token must not affect earlier positions' output.
    This verifies the causal mask is working.
    """
    torch.manual_seed(0)
    attn = Attention(d_model=64, n_heads=4, dropout=0.0)
    attn.eval()

    x = torch.randn(1, 6, 64)
    x2 = x.clone()
    x2[0, -1, :] = torch.randn(64)   # modify only the last token

    out1 = attn(x)
    out2 = attn(x2)

    # Position 0 should see identical outputs — it cannot attend to position 5
    assert torch.allclose(out1[0, 0], out2[0, 0], atol=1e-5), (
        "First token output changed when last token was modified — causal mask broken!"
    )

def test_attention_n_heads_must_divide_d_model():
    with pytest.raises(AssertionError):
        Attention(d_model=64, n_heads=5)   # 64 % 5 != 0
