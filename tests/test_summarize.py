"""Unit tests for the summarization package (chunking and data prep).

Uses fake tokenizers so the tests are fast and need no model download.
Run:  python tests/test_summarize.py
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "summarization"))
import summarize
from summarize import chunk_by_tokens
from data import load_qmsum_pairs, strip_speakers


class FakeTokenizer:
    """Each whitespace-separated integer is one token."""

    def encode(self, text, add_special_tokens=False):
        return [int(x) for x in text.split()] if text.strip() else []

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(i) for i in ids)


class WordTokenizer:
    """Each whitespace-separated word is one token (works with any text)."""

    def encode(self, text, add_special_tokens=False):
        return text.split()

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(ids)


# chunk_by_tokens

def test_empty_text_gives_no_chunks():
    assert chunk_by_tokens("", FakeTokenizer(), max_tokens=4, overlap=1) == []


def test_short_text_is_single_chunk():
    assert chunk_by_tokens("0 1 2", FakeTokenizer(), max_tokens=4, overlap=1) == ["0 1 2"]


def test_long_text_is_split_with_overlap():
    # 10 tokens, window 4, overlap 1 -> step 3 -> starts at 0, 3, 6
    chunks = chunk_by_tokens("0 1 2 3 4 5 6 7 8 9", FakeTokenizer(), max_tokens=4, overlap=1)
    assert chunks == ["0 1 2 3", "3 4 5 6", "6 7 8 9"]


def test_every_token_is_covered():
    text = " ".join(str(i) for i in range(50))
    chunks = chunk_by_tokens(text, FakeTokenizer(), max_tokens=8, overlap=2)
    covered = set()
    for c in chunks:
        covered.update(int(x) for x in c.split())
    assert covered == set(range(50))


def test_no_overlap_partitions_exactly():
    chunks = chunk_by_tokens("0 1 2 3 4 5", FakeTokenizer(), max_tokens=2, overlap=0)
    assert chunks == ["0 1", "2 3", "4 5"]


# summarize_document (model stubbed)

def test_summarize_document_reduces_to_one_summary():
    original = summarize.summarize_chunk
    summarize.summarize_chunk = lambda text, *a, **k: "x"
    try:
        text = " ".join(["w"] * 20)
        out = summarize.summarize_document(text, model=None, tokenizer=WordTokenizer(),
                                           device="cpu", chunk_tokens=5)
        assert isinstance(out, str) and out.strip() != ""
    finally:
        summarize.summarize_chunk = original


# data prep

def test_strip_speakers_drops_labels():
    turns = [{"speaker": "Alice", "content": " hi "}, {"speaker": "Bob", "content": "yo"}]
    assert strip_speakers(turns) == "hi yo"


def _write_split(dir_, records):
    with open(os.path.join(dir_, "train.jsonl"), "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_load_qmsum_pairs_builds_input_and_target():
    rec = {"meeting_transcripts": [{"speaker": "A", "content": "hello there ."},
                                   {"speaker": "B", "content": "bye ."}],
           "general_query_list": [{"query": "Summarize the whole meeting.", "answer": "a summary"}]}
    with tempfile.TemporaryDirectory() as d:
        _write_split(d, [rec])
        pairs = load_qmsum_pairs("train", data_dir=d)
    assert len(pairs) == 1
    assert pairs[0]["target"] == "a summary"
    assert pairs[0]["input"] == "Summarize the whole meeting. hello there . bye ."


def test_load_qmsum_pairs_skips_empty_answers():
    rec = {"meeting_transcripts": [{"speaker": "A", "content": "x"}],
           "general_query_list": [{"query": "q", "answer": "   "}]}
    with tempfile.TemporaryDirectory() as d:
        _write_split(d, [rec])
        assert load_qmsum_pairs("train", data_dir=d) == []


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\n{len(tests)} tests passed.")
