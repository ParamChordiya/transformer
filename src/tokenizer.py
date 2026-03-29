import tiktoken


class Tokenizer:
    def __init__(self):
        self._enc = tiktoken.get_encoding("gpt2")
        self.vocab_size = self._enc.n_vocab  # 50257

    def encode(self, text: str) -> list[int]:
        return self._enc.encode(text)

    def decode(self, tokens: list[int]) -> str:
        return self._enc.decode(tokens)

    def encode_file(self, path: str) -> list[int]:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        return self.encode(text)
