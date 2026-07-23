"""
Evaluate the trained BiLSTM segmenter on the QMSum test split.

Loads the checkpoint from train_segmenter.py (weights + the decode threshold
chosen on validation) and reports Pk / WindowDiff on test - the numbers for the
report - alongside the never-split baseline for context.

    python src/evaluate_segmenter.py

Needs the test cache (src/embed_cache.py --split test).
"""
import argparse
import os

import torch

from segmenter import BiLSTMSegmenter
from seg_metrics import mean_scores

CACHE_DIR = "data/seg_cache"
CKPT = "models/segmenter.pt"


def load_cache(split):
    path = os.path.join(CACHE_DIR, f"{split}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found - run: python src/embed_cache.py --split {split}")
    return torch.load(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--ckpt", default=CKPT, help="checkpoint to evaluate")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"{args.ckpt} not found - run: python src/train_segmenter.py")
    ckpt = torch.load(args.ckpt, map_location=args.device)
    model = BiLSTMSegmenter(hidden=ckpt["hidden"]).to(args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    thr = ckpt["threshold"]

    data = load_cache(args.split)
    model_pairs, ns_pairs = [], []
    for m in data:
        with torch.no_grad():
            logits = model(m["emb"].unsqueeze(0).to(args.device)).squeeze(0)
            probs = torch.sigmoid(logits).cpu().numpy()
        hyp = (probs > thr).astype(int)
        hyp[0] = 1
        labels = m["labels"].tolist()
        model_pairs.append((labels, hyp.tolist()))
        ns = [0] * len(labels)
        ns[0] = 1
        ns_pairs.append((labels, ns))

    ns_pk, ns_wd = mean_scores(ns_pairs)
    m_pk, m_wd = mean_scores(model_pairs)

    print(f"\n{args.split} split ({len(data)} meetings), decode threshold={thr}")
    print(f"  never-split baseline   Pk={ns_pk:.3f}  WindowDiff={ns_wd:.3f}")
    print(f"  BiLSTM segmenter       Pk={m_pk:.3f}  WindowDiff={m_wd:.3f}")


if __name__ == "__main__":
    main()
