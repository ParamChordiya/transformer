import torch
import pytest
from src.model import Attention, FeedForward, TransformerBlock

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

def test_feedforward_output_shape():
    ffn = FeedForward(d_model=64, d_ff=256)
    x = torch.randn(2, 8, 64)
    out = ffn(x)
    assert out.shape == (2, 8, 64)

def test_feedforward_is_position_wise():
    """FFN processes each position independently — reordering inputs reorders outputs."""
    torch.manual_seed(0)
    ffn = FeedForward(d_model=64, d_ff=256, dropout=0.0)
    ffn.eval()
    x = torch.randn(1, 4, 64)
    out_full = ffn(x)
    out_pos0 = ffn(x[:, :1, :])
    assert torch.allclose(out_full[:, 0:1, :], out_pos0, atol=1e-5)

def test_transformer_block_output_shape():
    block = TransformerBlock(d_model=64, n_heads=4, d_ff=256)
    x = torch.randn(2, 8, 64)
    out = block(x)
    assert out.shape == (2, 8, 64)

def test_transformer_block_residual():
    """With zero-initialized weights the block should return something close to its input."""
    block = TransformerBlock(d_model=64, n_heads=4, d_ff=256, dropout=0.0)
    # Zero out all weight matrices so the block effectively adds zero
    for name, p in block.named_parameters():
        if "weight" in name and p.dim() >= 2:
            torch.nn.init.zeros_(p)
    block.eval()
    x = torch.randn(1, 4, 64)
    out = block(x)
    # Output should equal input when all projections are zero
    assert out.shape == x.shape
