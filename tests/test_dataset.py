import os
import torch
from src.dataset import TokenDataset, make_dataloader, show_batch, download_wikitext2
from src.tokenizer import Tokenizer

def test_token_dataset_length():
    tokens = list(range(100))
    ds = TokenDataset(tokens, context_length=10)
    assert len(ds) == 90  # 100 - context_length

def test_token_dataset_item_shapes():
    tokens = list(range(100))
    ds = TokenDataset(tokens, context_length=10)
    x, y = ds[0]
    assert x.shape == (10,)
    assert y.shape == (10,)

def test_token_dataset_offset():
    """y should be x shifted by 1 — this is the next-token prediction setup."""
    tokens = list(range(20))
    ds = TokenDataset(tokens, context_length=5)
    x, y = ds[0]
    assert x.tolist() == [0, 1, 2, 3, 4]
    assert y.tolist() == [1, 2, 3, 4, 5]

def test_make_dataloader_batch_shape():
    tokens = list(range(200))
    loader = make_dataloader(tokens, context_length=10, batch_size=4, shuffle=False)
    x, y = next(iter(loader))
    assert x.shape == (4, 10)
    assert y.shape == (4, 10)

def test_make_dataloader_dtype():
    tokens = list(range(200))
    loader = make_dataloader(tokens, context_length=10, batch_size=4, shuffle=False)
    x, y = next(iter(loader))
    assert x.dtype == torch.long
    assert y.dtype == torch.long

def test_show_batch_runs(capsys):
    tok = Tokenizer()
    tokens = tok.encode("The transformer model processes sequences of tokens and learns to predict the next one.")
    loader = make_dataloader(tokens, context_length=8, batch_size=2, shuffle=False)
    x, y = next(iter(loader))
    show_batch(x, y, tok)
    captured = capsys.readouterr()
    assert "Input:" in captured.out
    assert "Target:" in captured.out


def test_download_wikitext2_cache_hit(tmp_path):
    """When cached .txt files already exist, download_wikitext2 returns correct paths without re-downloading."""
    # Pre-create the cached text files (simulating already-downloaded state)
    for split in ["train", "valid", "test"]:
        (tmp_path / f"{split}.txt").write_text("fake text")

    # Call the function — should not call HuggingFace, should return correct paths
    paths = download_wikitext2(str(tmp_path))

    assert set(paths.keys()) == {"train", "valid", "test"}
    assert all(os.path.exists(p) for p in paths.values())
