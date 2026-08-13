"""
Split a raw transcript into topic sections using the trained segmenter. This is
the inference step used by the pipeline and the ASR-error analysis.

Usage:
    python src/segmentation/segment_transcript.py <transcript.txt>
"""
import argparse
import os
import re

import numpy as np
import torch

from embed import Embedder, pick_device
from model import BiLSTMSegmenter

CKPT = "models/segmenter.pt"


def split_sentences(text):
    """nltk sentence split, with a regex fallback if punkt isn't available."""
    try:
        from nltk.tokenize import sent_tokenize
        try:
            return sent_tokenize(text)
        except LookupError:
            import nltk
            nltk.download("punkt_tab", quiet=True)
            nltk.download("punkt", quiet=True)
            return sent_tokenize(text)
    except Exception:
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        return [p for p in parts if p]


def decode_boundaries(probs, target_len=40, num_sections=None, min_gap=3,
                      threshold=None):
    """Sorted boundary sentence indices (0 is always in there).

    In threshold mode, every position whose prob beats `threshold` is a
    boundary. In target-count mode (the default) we split the meeting into
    ~num_sections evenly spaced windows (num_sections defaults to len /
    target_len) and snap each boundary to the highest-prob sentence inside its
    window. That keeps sections roughly balanced while still letting the model
    pick where each split lands, and it falls back to even splits when the model
    has little signal (e.g. a rambly meeting with no clear topic markers).
    """
    probs = np.asarray(probs)
    n = len(probs)
    if threshold is not None:
        return [0] + [i for i in range(1, n) if probs[i] > threshold]
    if num_sections is None:
        num_sections = max(2, round(n / max(1, target_len)))
    num_sections = max(1, min(num_sections, n))
    if num_sections == 1:
        return [0]
    seg = n / num_sections
    half = max(min_gap, int(seg // 2))
    bounds = [0]
    for k in range(1, num_sections):
        ideal = int(round(k * seg))
        lo = max(bounds[-1] + min_gap, ideal - half)
        hi = min(n, ideal + half)
        if lo >= hi:
            continue
        cand = lo + int(probs[lo:hi].argmax())
        if cand > bounds[-1]:
            bounds.append(cand)
    return bounds


def segment_text(text, model, embedder, device, target_len=40,
                 num_sections=None, min_gap=3, threshold=None, units=None):
    """transcript text -> (sections, units, boundary_indices).

    `units` lets the caller supply its own unit list (e.g. fixed word-windows
    for the ASR-error comparison); otherwise the text is split into sentences.
    """
    sents = units if units is not None else split_sentences(text)
    if len(sents) <= 1:
        return [text.strip()], sents, [0]
    emb = embedder.encode(sents)
    with torch.no_grad():
        logits = model(emb.unsqueeze(0).to(device)).squeeze(0)
        probs = torch.sigmoid(logits).cpu().numpy()
    bounds = decode_boundaries(probs, target_len, num_sections, min_gap, threshold)
    sections = []
    for j, start in enumerate(bounds):
        end = bounds[j + 1] if j + 1 < len(bounds) else len(sents)
        sections.append(" ".join(sents[start:end]).strip())
    return sections, sents, bounds


def load_model(ckpt, device):
    c = torch.load(ckpt, map_location=device)
    model = BiLSTMSegmenter(hidden=c["hidden"]).to(device)
    model.load_state_dict(c["model"])
    model.eval()
    return model


def segment_transcript(text, ckpt=CKPT, device=None, **kw):
    """Convenience wrapper for other code (e.g. the summarizer): returns just
    the list of section strings."""
    device = pick_device(device)
    model = load_model(ckpt, device)
    embedder = Embedder(device)
    sections, _, _ = segment_text(text, model, embedder, device, **kw)
    return sections


def main():
    ap = argparse.ArgumentParser(description="Segment a transcript into topic sections.")
    ap.add_argument("transcript", help="path to a transcript .txt file")
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--device", default=None, help="cpu | mps | cuda (default: auto)")
    ap.add_argument("--target-len", type=int, default=40,
                    help="aim for ~1 section per this many sentences")
    ap.add_argument("--num-sections", type=int, default=None,
                    help="force an exact number of sections (overrides --target-len)")
    ap.add_argument("--min-gap", type=int, default=3,
                    help="minimum sentences between boundaries")
    ap.add_argument("--threshold", type=float, default=None,
                    help="use probability-threshold decoding instead of target-count")
    ap.add_argument("--preview", type=int, default=100, help="chars shown per section")
    ap.add_argument("--full", action="store_true", help="print full section text")
    args = ap.parse_args()

    if not os.path.exists(args.transcript):
        raise FileNotFoundError(args.transcript)
    with open(args.transcript, encoding="utf-8") as f:
        text = f.read().strip()

    device = pick_device(args.device)
    model = load_model(args.ckpt, device)
    embedder = Embedder(device)

    sections, sents, bounds = segment_text(
        text, model, embedder, device,
        target_len=args.target_len, num_sections=args.num_sections,
        min_gap=args.min_gap, threshold=args.threshold)

    print(f"{os.path.basename(args.transcript)}: {len(sents)} sentences -> "
          f"{len(sections)} sections (boundaries at sentences {bounds})\n")
    for i, start in enumerate(bounds):
        end = bounds[i + 1] - 1 if i + 1 < len(bounds) else len(sents) - 1
        body = sections[i] if args.full else sections[i][:args.preview].replace("\n", " ") + "…"
        print(f"── Section {i+1}  (sentences {start}-{end}, {end-start+1}) " + "─" * 18)
        print(body)
        print()


if __name__ == "__main__":
    main()
