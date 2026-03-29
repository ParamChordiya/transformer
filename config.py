from dataclasses import dataclass

@dataclass
class Config:
    # model
    vocab_size: int = 50257       # tiktoken GPT-2 BPE vocab size
    context_length: int = 256     # max sequence length
    d_model: int = 256            # embedding dimension
    n_heads: int = 8              # attention heads (d_model must be divisible by n_heads)
    n_layers: int = 6             # number of transformer blocks
    d_ff: int = 1024              # feed-forward inner dim (typically 4 * d_model)
    dropout: float = 0.1

    # training
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    max_epochs: int = 10
    warmup_steps: int = 200       # linear LR warmup before cosine decay
    grad_clip: float = 1.0
    eval_interval: int = 200      # print loss every N steps
    checkpoint_interval: int = 1000

    # paths
    data_dir: str = "data/wikitext2"
    checkpoint_dir: str = "checkpoints"
