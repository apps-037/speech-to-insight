"""
Unsupervised baselines for topic segmentation, to compare against the trained
BiLSTM. Both run on the same cached MiniLM embeddings, so it's apples-to-apples.

  - never-split: predict a single segment (no boundaries). Deceptively strong on
    QMSum because boundaries are rare (~1% of turns).
  - block-similarity: a TextTiling-style method. For each gap between turns,
    compare the average embedding of the w turns before vs. the w turns after; a
    low cosine similarity means the topic likely shifted. We place a boundary
    where the dissimilarity exceeds a threshold tuned on validation - the same
    decode rule the trained model uses, so it's a fair comparison.

    python src/segmentation/baselines.py       # tune block-sim on val, report on test

Needs the val and test caches (src/segmentation/embed.py --split val / test).
"""
import argparse
import os

import numpy as np
import torch

from metrics import mean_scores

CACHE_DIR = "data/seg_cache"


def load_cache(split):
    path = os.path.join(CACHE_DIR, f"{split}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found - run: python src/segmentation/embed.py --split {split}")
    return torch.load(path)


def never_split(n):
    labels = [0] * n
    labels[0] = 1
    return labels


def block_similarity_scores(emb, w=5):
    """Per-gap dissimilarity 1 - cos(mean of w turns before, w turns after).

    Returns a [T] array; score[i] is the dissimilarity at the gap before turn i
    (higher = more likely a boundary). score[0] = inf so turn 0 always opens a
    segment, matching the model's decode.
    """
    e = emb.numpy() if hasattr(emb, "numpy") else np.asarray(emb)
    T = len(e)
    scores = np.zeros(T)
    scores[0] = np.inf
    for i in range(1, T):
        left = e[max(0, i - w):i].mean(0)
        right = e[i:i + w].mean(0)
        denom = np.linalg.norm(left) * np.linalg.norm(right) + 1e-9
        scores[i] = 1.0 - float(np.dot(left, right) / denom)
    return scores


def precompute(data, w):
    """[(labels, per-gap scores), ...] - computed once, then swept over."""
    return [(m["labels"].tolist(), block_similarity_scores(m["emb"], w))
            for m in data]


def eval_at(scored, thr):
    pairs = []
    for labels, scores in scored:
        hyp = (scores > thr).astype(int)
        hyp[0] = 1
        pairs.append((labels, hyp.tolist()))
    return mean_scores(pairs)


def tune_threshold(scored):
    best = (1e9, None, (None, None))
    for thr in np.arange(0.05, 0.80, 0.05):
        pk_, wd_ = eval_at(scored, float(thr))
        score = (pk_ + wd_) / 2
        if score < best[0]:
            best = (score, round(float(thr), 2), (round(pk_, 3), round(wd_, 3)))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=5)
    args = ap.parse_args()

    val = load_cache("val")
    test = load_cache("test")

    # never-split (no params, report directly on test)
    ns = mean_scores([(m["labels"].tolist(), never_split(len(m["labels"])))
                      for m in test])

    # block-similarity: tune threshold on val, apply to test
    _, thr, _ = tune_threshold(precompute(val, args.window))
    bs = eval_at(precompute(test, args.window), thr)

    print(f"\nBaselines on test ({len(test)} meetings):")
    print(f"  never-split                          Pk={ns[0]:.3f}  WindowDiff={ns[1]:.3f}")
    print(f"  block-similarity (w={args.window}, thr={thr})       "
          f"Pk={bs[0]:.3f}  WindowDiff={bs[1]:.3f}")


if __name__ == "__main__":
    main()
