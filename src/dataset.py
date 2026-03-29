import os
import torch
from torch.utils.data import Dataset, DataLoader

_HF_SPLITS = {"train": "train", "valid": "validation", "test": "test"}


def download_wikitext2(data_dir: str) -> dict[str, str]:
    """Download WikiText-2 via HuggingFace datasets and cache as plain text files."""
    from datasets import load_dataset  # lazy import — only needed once

    os.makedirs(data_dir, exist_ok=True)
    paths = {}
    for split, hf_split in _HF_SPLITS.items():
        cache_path = os.path.join(data_dir, f"{split}.txt")
        if not os.path.exists(cache_path):
            print(f"Downloading WikiText-2 ({split})...")
            ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=hf_split)
            with open(cache_path, "w", encoding="utf-8") as f:
                f.write("\n".join(ds["text"]))
        paths[split] = cache_path
    return paths


class TokenDataset(Dataset):
    def __init__(self, tokens: list[int], context_length: int):
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        self.context_length = context_length

    def __len__(self) -> int:
        return len(self.tokens) - self.context_length

    def __getitem__(self, idx: int):
        chunk = self.tokens[idx : idx + self.context_length + 1]
        x = chunk[:-1]   # input:  tokens[0..context_length-1]
        y = chunk[1:]    # target: tokens[1..context_length]  (shifted by 1)
        return x, y


def make_dataloader(
    tokens: list[int],
    context_length: int,
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:
    dataset = TokenDataset(tokens, context_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def show_batch(x: torch.Tensor, y: torch.Tensor, tokenizer) -> None:
    """Decode and print the first sample in a batch — useful for sanity-checking the data."""
    print("Input: ", tokenizer.decode(x[0].tolist()))
    print("Target:", tokenizer.decode(y[0].tolist()))
