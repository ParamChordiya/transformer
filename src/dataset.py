import os
import urllib.request
import zipfile
import torch
from torch.utils.data import Dataset, DataLoader

_WIKITEXT2_URL = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip"
_WIKITEXT2_SPLITS = {
    "train": "wikitext-2-raw/wiki.train.raw",
    "valid": "wikitext-2-raw/wiki.valid.raw",
    "test":  "wikitext-2-raw/wiki.test.raw",
}


def download_wikitext2(data_dir: str) -> dict[str, str]:
    """Download and extract WikiText-2 raw text. Returns paths keyed by split name."""
    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, "wikitext-2-raw-v1.zip")

    if not os.path.exists(zip_path):
        print("Downloading WikiText-2...")
        urllib.request.urlretrieve(_WIKITEXT2_URL, zip_path)

    # Extract once if any split file is missing
    if any(not os.path.exists(os.path.join(data_dir, p)) for p in _WIKITEXT2_SPLITS.values()):
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(data_dir)

    paths = {split: os.path.join(data_dir, rel_path) for split, rel_path in _WIKITEXT2_SPLITS.items()}
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
