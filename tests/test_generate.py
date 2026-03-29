import torch
from src.model import GPT
from src.tokenizer import Tokenizer
from src.generate import generate, load_from_checkpoint
from config import Config
import tempfile
import os

def _tiny_model():
    cfg = Config(vocab_size=50257, context_length=16, d_model=64, n_heads=4, n_layers=2, d_ff=256)
    return GPT(cfg), cfg

def test_generate_returns_string():
    model, _ = _tiny_model()
    tok = Tokenizer()
    result = generate(model, tok, prompt="The", max_new_tokens=5)
    assert isinstance(result, str)

def test_generate_extends_prompt():
    model, _ = _tiny_model()
    tok = Tokenizer()
    prompt = "Hello world"
    result = generate(model, tok, prompt=prompt, max_new_tokens=10)
    assert len(result) >= len(prompt)

def test_generate_deterministic_at_low_temperature():
    """Near-zero temperature makes sampling deterministic."""
    torch.manual_seed(42)
    model, _ = _tiny_model()
    model.eval()
    tok = Tokenizer()
    r1 = generate(model, tok, prompt="The", max_new_tokens=8, temperature=1e-10)
    r2 = generate(model, tok, prompt="The", max_new_tokens=8, temperature=1e-10)
    assert r1 == r2

def test_load_from_checkpoint():
    from src.train import save_checkpoint
    model, cfg = _tiny_model()
    optimizer = torch.optim.AdamW(model.parameters())
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg2 = Config(
            vocab_size=50257, context_length=16, d_model=64,
            n_heads=4, n_layers=2, d_ff=256, checkpoint_dir=tmpdir
        )
        save_checkpoint(model, optimizer, step=1, loss=3.0, cfg=cfg2)
        ckpt_path = os.path.join(tmpdir, "step_000001.pt")
        loaded = load_from_checkpoint(ckpt_path)
        assert isinstance(loaded, GPT)
        assert loaded.context_length == 16
