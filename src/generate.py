import torch
from src.model import GPT
from src.tokenizer import Tokenizer
from src.train import get_device


def generate(
    model: GPT,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    device: str | None = None,
) -> str:
    """
    Autoregressively sample tokens from the model.

    temperature: controls randomness.
      - 1.0 → sample from the raw distribution
      - < 1.0 → sharper, more predictable (→ 0 becomes greedy/argmax)
      - > 1.0 → flatter, more random
    """
    if device is None:
        device = get_device()
    model.eval()
    model.to(device)
    tokens = tokenizer.encode(prompt)
    x = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)  # (1, T)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Crop to context_length — the model can't process longer sequences
            x_crop = x[:, -model.context_length:]
            logits = model(x_crop)               # (1, T, vocab_size)
            logits = logits[:, -1, :] / temperature  # last position only: (1, vocab_size)
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)
            x = torch.cat([x, next_token], dim=1)

    return tokenizer.decode(x[0].tolist())


def load_from_checkpoint(path: str, device: str | None = None) -> GPT:
    """Load a GPT model from a checkpoint saved by src/train.py."""
    if device is None:
        device = get_device()
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    cfg = checkpoint["config"]
    model = GPT(cfg)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    return model
