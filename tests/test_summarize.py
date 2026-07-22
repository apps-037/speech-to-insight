"""Unit tests for the pure chunking logic in src/summarize.py.

Uses a fake tokenizer (token == whitespace-separated integer) so the test is
fast and needs no model download. Run:  python tests/test_summarize.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from summarize import chunk_by_tokens  # noqa: E402


class FakeTokenizer:
    """Minimal stand-in: each whitespace-separated integer is one token."""

    def encode(self, text, add_special_tokens=False):
        return [int(x) for x in text.split()] if text.strip() else []

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(i) for i in ids)


def test_empty_text_gives_no_chunks():
    tok = FakeTokenizer()
    assert chunk_by_tokens("", tok, max_tokens=4, overlap=1) == []


def test_short_text_is_single_chunk():
    tok = FakeTokenizer()
    chunks = chunk_by_tokens("0 1 2", tok, max_tokens=4, overlap=1)
    assert chunks == ["0 1 2"]


def test_long_text_is_split_with_overlap():
    tok = FakeTokenizer()
    # 10 tokens, window 4, overlap 1 -> step 3 -> starts at 0,3,6
    chunks = chunk_by_tokens("0 1 2 3 4 5 6 7 8 9", tok, max_tokens=4, overlap=1)
    assert chunks == ["0 1 2 3", "3 4 5 6", "6 7 8 9"]


def test_every_token_is_covered():
    tok = FakeTokenizer()
    text = " ".join(str(i) for i in range(50))
    chunks = chunk_by_tokens(text, tok, max_tokens=8, overlap=2)
    covered = set()
    for c in chunks:
        covered.update(int(x) for x in c.split())
    assert covered == set(range(50))


def test_no_overlap_partitions_exactly():
    tok = FakeTokenizer()
    chunks = chunk_by_tokens("0 1 2 3 4 5", tok, max_tokens=2, overlap=0)
    assert chunks == ["0 1", "2 3", "4 5"]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\n{len(tests)} tests passed.")
