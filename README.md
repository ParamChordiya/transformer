# Transformer from Scratch

A decoder-only GPT-style transformer built from scratch in PyTorch, trained on WikiText-2. Every component — attention, feed-forward, positional embeddings, training loop, and text generation — is implemented without abstraction libraries so you can see exactly how it works.

---

## What Was Built

### Architecture

A **decoder-only transformer** (the same family as GPT-2, LLaMA, Mistral). Given a sequence of tokens, it predicts the next token at every position simultaneously — this is called autoregressive language modeling.

```
token IDs
  → token embeddings + positional embeddings
  → dropout
  → N × TransformerBlock
       ├── LayerNorm (pre-norm)
       ├── Multi-Head Causal Self-Attention  ← can only look backwards
       ├── residual connection
       ├── LayerNorm (pre-norm)
       ├── Feed-Forward Network (Linear → GELU → Linear)
       └── residual connection
  → final LayerNorm
  → linear head → logits over vocabulary (50,257 tokens)
```

**Default model size:** ~30.5M parameters

| Hyperparameter | Default | Meaning |
|---|---|---|
| `vocab_size` | 50,257 | GPT-2 BPE vocabulary |
| `context_length` | 256 | Max tokens the model sees at once |
| `d_model` | 256 | Embedding dimension |
| `n_heads` | 8 | Attention heads (d_head = 32 each) |
| `n_layers` | 6 | Transformer blocks stacked |
| `d_ff` | 1,024 | Feed-forward inner dimension (4× d_model) |
| `dropout` | 0.1 | Dropout rate |

All hyperparameters live in `config.py` — edit one file to change the whole model.

### Key Design Choices

- **Pre-norm** — LayerNorm is applied *before* each sub-layer (not after), which is the modern convention used by GPT-2 onward and trains more stably.
- **Causal mask** — An upper-triangular mask prevents each token from attending to future positions. Pre-allocated as a PyTorch buffer so it moves to GPU automatically.
- **Fused QKV** — Query, Key, and Value projections are computed in a single linear layer and then split, which is more efficient than three separate layers.
- **AdamW + cosine LR** — Linear warmup for the first `warmup_steps` steps, then cosine decay down to `min_lr`.
- **tiktoken BPE** — Uses the same tokenizer as GPT-2/GPT-4. "transformer" → 2 subword tokens.
- **MPS support** — Automatically uses Apple Silicon GPU (Metal) when available, falls back to CUDA then CPU.

---

## Project Structure

```
transformer/
├── config.py              # All hyperparameters in one dataclass
├── requirements.txt       # Dependencies
│
├── src/
│   ├── tokenizer.py       # tiktoken GPT-2 BPE wrapper
│   ├── dataset.py         # WikiText-2 download + TokenDataset + DataLoader
│   ├── model.py           # Attention → FeedForward → TransformerBlock → GPT
│   ├── train.py           # Training loop: AdamW, cosine LR, checkpointing
│   └── generate.py        # Temperature sampling, checkpoint loading
│
├── notebooks/
│   └── explore.ipynb      # Interactive: inspect batches, attention viz, generation
│
├── tests/                 # 31 unit tests (pytest)
│   ├── test_tokenizer.py
│   ├── test_dataset.py
│   ├── test_model.py
│   ├── test_train.py
│   └── test_generate.py
│
├── data/                  # WikiText-2 downloaded here on first run
└── checkpoints/           # Saved model weights go here
```

---

## Setup

```bash
# Clone the repo
git clone https://github.com/ParamChordiya/transformer.git
cd transformer

# Install dependencies
pip install -r requirements.txt
```

---

## Commands

### Train the model

```bash
PYTHONPATH=. python -u src/train.py
```

On first run this downloads WikiText-2 (~5MB) via HuggingFace datasets. Training then starts printing loss and perplexity every 200 steps. Checkpoints are saved to `checkpoints/` every 1,000 steps.

**What you'll see:**
```
Downloading WikiText-2 (train)...
Downloading WikiText-2 (valid)...
Downloading WikiText-2 (test)...
Model parameters: 30,530,048
  step    200 | loss 6.2341 | ppl 507.8 | lr 2.76e-04
  step    400 | loss 5.8102 | ppl 333.5 | lr 2.53e-04
  ...
```

Perplexity starts high (random model ≈ 50k) and drops as the model learns. A well-trained run reaches perplexity in the low hundreds after a few epochs.

**Device selection** is automatic: Apple MPS → CUDA → CPU.

### Run the test suite

```bash
pytest -v
```

31 tests covering every component: tokenizer roundtrip, dataset offset correctness, causal mask behavioral test, exact parameter count, training smoke test, generation determinism.

### Generate text from a checkpoint

```python
from src.generate import generate, load_from_checkpoint
from src.tokenizer import Tokenizer

model = load_from_checkpoint("checkpoints/step_001000.pt")
tok = Tokenizer()

# temperature < 1.0 → more focused, temperature > 1.0 → more random
print(generate(model, tok, prompt="The history of", max_new_tokens=100, temperature=0.8))
```

### Inspect a training batch interactively

```python
from src.tokenizer import Tokenizer
from src.dataset import download_wikitext2, make_dataloader, show_batch

tok = Tokenizer()
paths = download_wikitext2("data/wikitext2")
tokens = tok.encode_file(paths["train"])
loader = make_dataloader(tokens, context_length=32, batch_size=4)
x, y = next(iter(loader))
show_batch(x, y, tok)
# Input:  The transformer model learns to predict
# Target: transformer model learns to predict the
```

### Check model parameter count

```python
from config import Config
from src.model import GPT

cfg = Config()
model = GPT(cfg)
print(f"Parameters: {model.num_params():,}")

for name, module in model.named_children():
    n = sum(p.numel() for p in module.parameters())
    print(f"  {name}: {n:,}")
```

### Try a smaller/faster model

Edit `config.py` or override at runtime:

```python
from config import Config
from src.model import GPT
from src.train import train

cfg = Config(
    d_model=128,
    n_heads=4,
    n_layers=4,
    d_ff=512,
    context_length=128,
    max_epochs=3,
)
train(cfg)
```

This gives a ~7M parameter model that trains much faster — useful for quick experiments.

---

## How the Components Work

### `src/tokenizer.py` — Tokenizer
Thin wrapper around `tiktoken`. BPE (Byte Pair Encoding) splits words into subword units: "transformer" → `["transform", "er"]`. This lets the model handle unknown words gracefully and keeps the vocabulary at a manageable ~50k tokens.

### `src/dataset.py` — TokenDataset
Downloads WikiText-2 and wraps it in a PyTorch `Dataset`. Each sample is a pair `(x, y)` where `y = x` shifted one position right — this is the next-token prediction objective. Every forward pass trains the model on `context_length` predictions simultaneously.

### `src/model.py` — The Model (read bottom-up)

**`Attention`** — Multi-head causal self-attention. Each token builds a weighted average of all previous tokens' values, where weights come from how similar the token's query is to each past token's key. The upper-triangular causal mask ensures position `i` only attends to positions `0..i`.

**`FeedForward`** — Applied independently to each position after attention. Two linear layers with GELU activation: expands to 4× the embedding dimension then projects back. This is where most of the model's "memory" lives.

**`TransformerBlock`** — Wraps attention and FFN with pre-norm and residual connections. The residual connections (`x = x + sublayer(ln(x))`) let gradients flow directly to early layers, enabling training of deep networks.

**`GPT`** — Stacks N blocks on top of token + positional embeddings. The positional embeddings are learned (not sinusoidal), giving the model a sense of where each token is in the sequence.

### `src/train.py` — Training Loop
Explicit PyTorch training loop — no Trainer abstraction. Cosine learning rate schedule: linearly warms up for `warmup_steps` steps (stabilises early training), then decays following a cosine curve. Gradient clipping (`max_norm=1.0`) prevents exploding gradients.

### `src/generate.py` — Text Generation
Autoregressive sampling: feed a prompt, take the last position's logits, divide by temperature, sample the next token, append and repeat. Temperature controls the sharpness of the distribution — lower temperature makes the model more confident and repetitive, higher makes it more creative and random.

---

## Configuration Reference

All settings in `config.py`:

| Parameter | Default | Description |
|---|---|---|
| `vocab_size` | 50,257 | Must match the tokenizer |
| `context_length` | 256 | Tokens per training sample |
| `d_model` | 256 | Embedding + hidden dimension |
| `n_heads` | 8 | Attention heads (must divide d_model) |
| `n_layers` | 6 | Transformer blocks |
| `d_ff` | 1,024 | Feed-forward inner dimension |
| `dropout` | 0.1 | Regularization |
| `batch_size` | 32 | Samples per gradient step |
| `learning_rate` | 3e-4 | Peak LR after warmup |
| `weight_decay` | 0.1 | AdamW weight decay |
| `max_epochs` | 10 | Training epochs |
| `warmup_steps` | 200 | Linear LR warmup steps |
| `grad_clip` | 1.0 | Max gradient norm |
| `eval_interval` | 200 | Print loss every N steps |
| `checkpoint_interval` | 1,000 | Save checkpoint every N steps |
| `data_dir` | `data/wikitext2` | Where to cache the dataset |
| `checkpoint_dir` | `checkpoints` | Where to save model weights |
