"""
Summarize a meeting transcript.

Reads a plain-text transcript (as produced by src/transcribe.py: one text blob,
no speaker labels or timestamps) and writes an abstractive summary to
data/summaries/<name>.txt.

Meeting transcripts are far longer than the model's input limit, so we use a
map-reduce strategy: split the transcript into token-bounded chunks, summarize
each chunk, then recursively summarize the joined chunk-summaries until a single
overall summary remains.

The chunker (`chunk_by_tokens`) is a pure function so it can be unit-tested and
so a topic-segmentation boundary function can replace it later without changing
the rest of the code.

Usage:
    python src/summarize.py data/transcripts/EN2001a.txt
    python src/summarize.py data/transcripts/EN2001a.txt --model <hf-name-or-path>
"""
import argparse
import os
import sys

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.utils import logging as hf_logging

# We deliberately encode the full (very long) transcript only to split it into
# chunks; the model itself only ever sees chunk-sized input. Silence the
# resulting "sequence longer than 1024" tokenizer warning to keep output clean.
hf_logging.set_verbosity_error()

DEFAULT_MODEL = "sshleifer/distilbart-cnn-12-6"


def chunk_by_tokens(text, tokenizer, max_tokens=800, overlap=50):
    """Split `text` into chunks of at most `max_tokens` tokens.

    Returns a list of decoded string chunks with a small token overlap between
    consecutive chunks. Pure function of (text, tokenizer) — unit-testable.
    """
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        return []
    step = max(1, max_tokens - overlap)
    chunks = []
    start = 0
    n = len(ids)
    while start < n:
        window = ids[start:start + max_tokens]
        chunks.append(tokenizer.decode(window, skip_special_tokens=True))
        if start + max_tokens >= n:
            break
        start += step
    return chunks


def summarize_chunk(text, model, tokenizer, device, max_input=1024,
                    max_summary=160, min_summary=30):
    """Summarize a single chunk that already fits (roughly) in the model."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       max_length=max_input).to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_length=max_summary,
            min_length=min_summary,
            num_beams=4,
            length_penalty=2.0,
            no_repeat_ngram_size=3,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True).strip()


def summarize_document(text, model, tokenizer, device, chunk_tokens=800,
                       max_input=1024, max_summary=160, min_summary=30):
    """Map-reduce summarization for arbitrarily long text.

    Summarize each chunk (map), then recurse on the joined summaries (reduce)
    until everything fits in one chunk. Always converges because each pass is
    much shorter than its input.
    """
    chunks = chunk_by_tokens(text, tokenizer, max_tokens=chunk_tokens)
    if len(chunks) <= 1:
        return summarize_chunk(chunks[0] if chunks else text, model, tokenizer,
                               device, max_input, max_summary, min_summary)
    partials = [
        summarize_chunk(c, model, tokenizer, device, max_input, max_summary, min_summary)
        for c in chunks
    ]
    print(f"  summarized {len(chunks)} chunks; reducing...")
    return summarize_document(" ".join(partials), model, tokenizer, device,
                              chunk_tokens, max_input, max_summary, min_summary)


def load_model(model_name, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    model.eval()
    return model, tokenizer


def main():
    ap = argparse.ArgumentParser(description="Summarize a meeting transcript.")
    ap.add_argument("transcript", help="path to a transcript .txt file")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="HuggingFace model name or local checkpoint path")
    ap.add_argument("--out", default=None, help="output path (default data/summaries/<name>.txt)")
    ap.add_argument("--max-summary", type=int, default=160)
    ap.add_argument("--min-summary", type=int, default=30)
    ap.add_argument("--device", default=None, help="cpu | mps | cuda (default: auto)")
    args = ap.parse_args()

    with open(args.transcript, encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        print("Transcript is empty.")
        sys.exit(1)

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Loading model '{args.model}' on {device}...")
    model, tokenizer = load_model(args.model, device)

    print("Summarizing...")
    summary = summarize_document(text, model, tokenizer, device,
                                 max_summary=args.max_summary,
                                 min_summary=args.min_summary)

    out_path = args.out or f"data/summaries/{os.path.splitext(os.path.basename(args.transcript))[0]}.txt"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(summary + "\n")

    print("\n=== SUMMARY ===")
    print(summary)
    print(f"\nSaved summary to {out_path}")


if __name__ == "__main__":
    main()
