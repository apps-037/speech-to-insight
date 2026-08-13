"""
Train the BiLSTM segmenter on the cached QMSum embeddings and save the best
checkpoint to models/segmenter.pt.

Usage:
    python src/segmentation/train.py
"""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn

from model import BiLSTMSegmenter
from metrics import mean_scores

CACHE_DIR = "data/seg_cache"
CKPT = "models/segmenter.pt"


def load_cache(split):
    path = os.path.join(CACHE_DIR, f"{split}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found - run: python src/segmentation/embed.py --split {split}")
    return torch.load(path)


def predict_probs(model, emb, device):
    with torch.no_grad():
        logits = model(emb.unsqueeze(0).to(device)).squeeze(0)
        return torch.sigmoid(logits).cpu().numpy()


def val_predictions(model, data, device):
    """Run the model once over `data`; return [(labels, probs), ...]. Threshold
    tuning then sweeps over these cached probs instead of re-running the model."""
    model.eval()
    return [(m["labels"].tolist(), predict_probs(model, m["emb"], device))
            for m in data]


def _score(labels, probs, thr):
    hyp = (probs > thr).astype(int)
    hyp[0] = 1                          # first unit always opens a segment
    return labels, hyp.tolist()


def eval_at(preds, thr):
    return mean_scores(_score(lab, p, thr) for lab, p in preds)


def best_threshold(preds):
    """Sweep thresholds over precomputed predictions; return
    (score, thr, (Pk, WindowDiff)) minimizing the mean of Pk and WindowDiff."""
    best = (1e9, None, (None, None))
    for thr in np.arange(0.30, 0.96, 0.05):
        pk_, wd_ = eval_at(preds, float(thr))
        score = (pk_ + wd_) / 2
        if score < best[0]:
            best = (score, round(float(thr), 2), (round(pk_, 3), round(wd_, 3)))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--max-pos-weight", type=float, default=5.0,
                    help="cap on n_neg/n_pos; raw value (~30-50) over-segments")
    ap.add_argument("--out", default=CKPT, help="checkpoint output path")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    device = args.device
    train = load_cache("train")
    val = load_cache("val")
    print(f"train: {len(train)} meetings, val: {len(val)}", flush=True)

    pos = sum(int(m["labels"].sum()) for m in train)
    tot = sum(len(m["labels"]) for m in train)
    raw_pw = (tot - pos) / max(1, pos)
    pw = min(raw_pw, args.max_pos_weight)
    print(f"boundary rate={pos / tot:.3f}  raw pos_weight={raw_pw:.1f}  "
          f"using {pw:.1f}", flush=True)

    model = BiLSTMSegmenter(hidden=args.hidden).to(device)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw], device=device))
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    base_pk, base_wd = eval_at(val_predictions(model, val, device), thr=2.0)
    print(f"never-split baseline: Pk={base_pk:.3f} WindowDiff={base_wd:.3f}\n",
          flush=True)

    best_score, best_state = 1e9, None
    order = list(range(len(train)))
    for ep in range(args.epochs):
        model.train()
        np.random.shuffle(order)
        for i in order:
            m = train[i]
            emb = m["emb"].unsqueeze(0).to(device)
            lab = m["labels"].float().unsqueeze(0).to(device)
            opt.zero_grad()
            loss = crit(model(emb), lab)
            loss.backward()
            opt.step()

        score, thr, (pk_, wd_) = best_threshold(val_predictions(model, val, device))
        flag = ""
        if score < best_score:
            best_score = score
            best_state = {"model": model.state_dict(), "threshold": thr,
                          "hidden": args.hidden}
            flag = "  *"
        print(f"epoch {ep:2}: thr={thr}  Pk={pk_}  WindowDiff={wd_}{flag}",
              flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(best_state, args.out)
    print(f"\nSaved best checkpoint (val score {best_score:.3f}) to {args.out}",
          flush=True)


if __name__ == "__main__":
    main()
