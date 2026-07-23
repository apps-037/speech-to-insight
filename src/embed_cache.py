"""
One-time step: embed every QMSum turn with a frozen sentence encoder and cache
the vectors to disk.

We load all-MiniLM-L6-v2 directly through `transformers` (mean-pool the token
embeddings over the attention mask, then L2-normalize - exactly what
sentence-transformers does under the hood). This avoids sentence-transformers'
torchcodec/FFmpeg dependency, which breaks on this setup. The output is
identical: 384-d normalized sentence vectors.

This is the only GPU-worth-it part of segmentation. Run it once (locally on
Apple-Silicon MPS, or on a Colab GPU and download the cache), then every other
script trains/evaluates on the cached vectors in seconds on CPU.

    python src/embed_cache.py --split train
    python src/embed_cache.py --split val
    python src/embed_cache.py --split test

Writes data/seg_cache/<split>.pt = list of dicts
    {"emb": FloatTensor[T, 384], "labels": LongTensor[T]}
one per meeting.
"""
import argparse
import os

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from qmsum_seg_prep import iter_labeled, DEFAULT_QMSUM_DIR

ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
CACHE_DIR = "data/seg_cache"


def pick_device(arg):
    if arg:
        return arg
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Embedder:
    """all-MiniLM-L6-v2 via transformers: mean-pool token embeddings over the
    attention mask, then L2-normalize (the sentence-transformers recipe)."""

    def __init__(self, device):
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(ENCODER)
        self.model = AutoModel.from_pretrained(ENCODER).to(device).eval()

    @torch.no_grad()
    def encode(self, texts, batch_size=64, max_length=256):
        out = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = self.tok(batch, padding=True, truncation=True,
                           max_length=max_length, return_tensors="pt").to(self.device)
            hidden = self.model(**enc).last_hidden_state          # [B, L, 384]
            mask = enc["attention_mask"].unsqueeze(-1).float()    # [B, L, 1]
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            pooled = F.normalize(pooled, p=2, dim=1)
            out.append(pooled.cpu())
        return torch.cat(out, 0)


def build_cache(split, qmsum_dir, embedder):
    meetings = []
    for texts, labels in iter_labeled(split, qmsum_dir):
        emb = embedder.encode(texts)
        meetings.append({
            "emb": emb.float(),
            "labels": torch.tensor(labels, dtype=torch.long),
        })
        print(f"  {split}: {len(meetings)} meetings", end="\r")
    print()
    return meetings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "val", "test"])
    ap.add_argument("--qmsum-dir", default=DEFAULT_QMSUM_DIR)
    ap.add_argument("--device", default=None,
                    help="cpu | mps | cuda (default: auto)")
    args = ap.parse_args()

    device = pick_device(args.device)
    print(f"Embedding '{args.split}' with {ENCODER} on {device}...")
    embedder = Embedder(device)
    meetings = build_cache(args.split, args.qmsum_dir, embedder)

    os.makedirs(CACHE_DIR, exist_ok=True)
    out = os.path.join(CACHE_DIR, f"{args.split}.pt")
    torch.save(meetings, out)
    print(f"Saved {len(meetings)} meetings to {out}")


if __name__ == "__main__":
    main()
