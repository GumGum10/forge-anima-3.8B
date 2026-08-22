from __future__ import annotations

from transformers import AutoTokenizer

from .files import tokenizer_dir


class Qwen35Tokenizer:
    pad_token_id = 151643
    max_length = 1024

    def __init__(self):
        self.inner = AutoTokenizer.from_pretrained(
            tokenizer_dir(), trust_remote_code=False, local_files_only=True
        )

    def __call__(self, texts, **kwargs):
        kwargs.setdefault("truncation", False)
        kwargs.setdefault("add_special_tokens", False)
        result = self.inner(texts, **kwargs)
        result["input_ids"] = [
            (tokens[: self.max_length] or [self.pad_token_id])
            for tokens in result["input_ids"]
        ]
        return result
