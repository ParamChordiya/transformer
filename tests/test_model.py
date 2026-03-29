import torch
import pytest
from src.model import Attention, FeedForward, TransformerBlock, GPT
from config import Config

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
    """Residual connections: if sublayer output is zero, block should return input unchanged."""
    torch.manual_seed(0)
    block = TransformerBlock(d_model=64, n_heads=4, d_ff=256, dropout=0.0)
    # Zero ALL parameters so both sublayers produce zero vectors
    with torch.no_grad():
        for p in block.parameters():
            p.zero_()
    block.eval()
    x = torch.randn(1, 4, 64)
    out = block(x)
    # With all weights and biases zero, sublayer outputs are zero → x + 0 = x
    assert torch.allclose(out, x, atol=1e-5), (
        f"Residual connection broken: max diff = {(out - x).abs().max().item()}"
    )

def test_gpt_output_shape():
    cfg = Config(vocab_size=100, context_length=16, d_model=64, n_heads=4, n_layers=2, d_ff=256)
    model = GPT(cfg)
    idx = torch.randint(0, 100, (2, 16))   # (batch=2, seq=16)
    logits = model(idx)
    assert logits.shape == (2, 16, 100)    # (batch, seq, vocab_size)

def test_gpt_num_params():
    cfg = Config(vocab_size=100, context_length=16, d_model=64, n_heads=4, n_layers=2, d_ff=256)
    model = GPT(cfg)
    # Computed: token_emb(6400) + pos_emb(1024) + 2 blocks(49728 each) + ln_f(128) + head(6400) = 113408
    assert model.num_params() == 113408

def test_gpt_context_length_stored():
    cfg = Config(vocab_size=100, context_length=32, d_model=64, n_heads=4, n_layers=2, d_ff=256)
    model = GPT(cfg)
    assert model.context_length == 32

def test_gpt_logits_vary_by_position():
    """Different positions should produce different logits — model is not position-agnostic."""
    torch.manual_seed(1)
    cfg = Config(vocab_size=100, context_length=16, d_model=64, n_heads=4, n_layers=2, d_ff=256)
    model = GPT(cfg)
    model.eval()
    idx = torch.randint(0, 100, (1, 8))
    logits = model(idx)
    # Logits at position 0 and position 7 should differ
    assert not torch.allclose(logits[0, 0], logits[0, 7])
