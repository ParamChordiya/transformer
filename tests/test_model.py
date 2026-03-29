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
    No position i should be affected by changes to any position j > i.
    This verifies the causal mask blocks all future information.
    """
    torch.manual_seed(0)
    attn = Attention(d_model=64, n_heads=4, dropout=0.0)
    attn.eval()

    x = torch.randn(1, 6, 64)

    for pos in range(5):  # modify each position in turn, check earlier ones are unaffected
        x2 = x.clone()
        x2[0, pos + 1, :] = torch.randn(64)  # modify position pos+1

        out1 = attn(x)
        out2 = attn(x2)

        for i in range(pos + 1):  # positions 0..pos must be unchanged
            assert torch.allclose(out1[0, i], out2[0, i], atol=1e-5), (
                f"Position {i} output changed when position {pos+1} was modified — causal mask broken!"
            )

def test_attention_n_heads_must_divide_d_model():
    with pytest.raises(AssertionError):
        Attention(d_model=64, n_heads=5)   # 64 % 5 != 0
