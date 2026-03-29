import math
import os
import torch
import torch.nn as nn

from config import Config
from src.model import GPT
from src.tokenizer import Tokenizer
from src.dataset import download_wikitext2, make_dataloader


def cosine_lr(
    step: int,
    warmup_steps: int,
    max_steps: int,
    max_lr: float,
    min_lr: float = 1e-5,
) -> float:
    """Linear warmup then cosine decay learning rate schedule."""
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


def save_checkpoint(model: GPT, optimizer: torch.optim.Optimizer, step: int, loss: float, cfg: Config) -> None:
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    path = os.path.join(cfg.checkpoint_dir, f"step_{step:06d}.pt")
    torch.save({
        "step": step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "loss": loss,
        "config": cfg,
    }, path)
    print(f"  checkpoint saved → {path}")


def train(cfg: Config, tokens_override: dict | None = None) -> list[float]:
    """
    Main training loop.

    tokens_override: if provided, use these token lists instead of downloading WikiText-2.
    Returns list of training losses (one per eval_interval).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if tokens_override is not None:
        train_tokens = tokens_override["train"]
        val_tokens = tokens_override["valid"]
    else:
        tokenizer = Tokenizer()
        paths = download_wikitext2(cfg.data_dir)
        train_tokens = tokenizer.encode_file(paths["train"])
        val_tokens = tokenizer.encode_file(paths["valid"])

    train_loader = make_dataloader(train_tokens, cfg.context_length, cfg.batch_size, shuffle=True)
    val_loader = make_dataloader(val_tokens, cfg.context_length, cfg.batch_size, shuffle=False)

    model = GPT(cfg).to(device)
    print(f"Model parameters: {model.num_params():,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    max_steps = len(train_loader) * cfg.max_epochs
    step = 0
    losses = []

    for epoch in range(cfg.max_epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            # Update learning rate according to schedule
            lr = cosine_lr(step, cfg.warmup_steps, max_steps, cfg.learning_rate)
            for g in optimizer.param_groups:
                g["lr"] = lr

            logits = model(x)                                          # (B, T, vocab_size)
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),                      # (B*T, vocab_size)
                y.view(-1),                                            # (B*T,)
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            step += 1

            if step % cfg.eval_interval == 0:
                ppl = math.exp(loss.item())
                print(f"  step {step:6d} | loss {loss.item():.4f} | ppl {ppl:.1f} | lr {lr:.2e}")
                losses.append(loss.item())

            if step % cfg.checkpoint_interval == 0:
                save_checkpoint(model, optimizer, step, loss.item(), cfg)

    return losses


if __name__ == "__main__":
    train(Config())
