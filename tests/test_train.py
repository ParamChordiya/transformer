import math
import torch
from src.train import cosine_lr, save_checkpoint, train
from config import Config
import os
import tempfile

def test_cosine_lr_warmup():
    """During warmup, LR should increase linearly from 0 to max_lr."""
    lr = cosine_lr(step=50, warmup_steps=100, max_steps=1000, max_lr=3e-4)
    assert abs(lr - 0.5 * 3e-4) < 1e-8   # halfway through warmup → half of max_lr

def test_cosine_lr_after_warmup_is_less_than_max():
    lr = cosine_lr(step=500, warmup_steps=100, max_steps=1000, max_lr=3e-4)
    assert lr < 3e-4

def test_cosine_lr_at_end_is_min():
    lr = cosine_lr(step=9999, warmup_steps=100, max_steps=1000, max_lr=3e-4, min_lr=1e-5)
    assert lr == 1e-5

def test_save_and_load_checkpoint():
    from src.model import GPT
    cfg = Config(vocab_size=100, context_length=16, d_model=32, n_heads=4, n_layers=1, d_ff=64)
    model = GPT(cfg)
    optimizer = torch.optim.AdamW(model.parameters())

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg2 = Config(checkpoint_dir=tmpdir)
        save_checkpoint(model, optimizer, step=1, loss=2.5, cfg=cfg2)
        ckpt_path = os.path.join(tmpdir, "step_000001.pt")
        assert os.path.exists(ckpt_path)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        assert ckpt["step"] == 1
        assert abs(ckpt["loss"] - 2.5) < 1e-6

def test_train_smoke():
    """3 training steps on random data — loss should be a finite number."""
    import tempfile
    cfg = Config(
        vocab_size=100,
        context_length=8,
        d_model=32,
        n_heads=4,
        n_layers=1,
        d_ff=64,
        batch_size=4,
        max_epochs=1,
        warmup_steps=1,
        eval_interval=1,
        checkpoint_interval=9999,
        checkpoint_dir=tempfile.mkdtemp(),
    )
    tokens = list(range(100)) * 10   # 1000 tokens of fake data
    losses = train(cfg, tokens_override={"train": tokens, "valid": tokens})
    assert len(losses) > 0
    assert all(math.isfinite(l) for l in losses)
