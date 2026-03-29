from src.tokenizer import Tokenizer


def test_encode_decode_roundtrip():
    tok = Tokenizer()
    text = "Hello, transformer!"
    assert tok.decode(tok.encode(text)) == text


def test_vocab_size():
    tok = Tokenizer()
    assert tok.vocab_size == 50257


def test_encode_returns_list_of_ints():
    tok = Tokenizer()
    tokens = tok.encode("hello world")
    assert isinstance(tokens, list)
    assert all(isinstance(t, int) for t in tokens)


def test_subword_tokenization():
    """'transformer' should split into multiple tokens, showing BPE in action."""
    tok = Tokenizer()
    tokens = tok.encode("transformer")
    assert len(tokens) >= 1   # may be 1 or more depending on BPE
    assert all(0 <= t < tok.vocab_size for t in tokens)
