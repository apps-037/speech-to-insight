"""
Evaluate the trained BiLSTM segmenter on the QMSum test split.

Loads the checkpoint from train.py (weights plus the decode threshold picked on
val) and prints Pk / WindowDiff on test, the numbers we report, next to the
never-split baseline for context.

    python src/segmentation/evaluate.py
    python src/segmentation/evaluate.py --save reports/metrics.json   # record for plots

Needs the test cache (src/segmentation/embed.py --split test).
"""
import argparse
import json
import os

import torch

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


def save_metrics(path, ns, bilstm, thr):
    """Read-modify-write the shared metrics JSON so the plots use real numbers.
    Only updates the segmentation model + never-split rows; leaves everything
    else (embedding-similarity baseline, ROUGE) untouched."""
    data = {}
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
    seg = data.setdefault("segmentation", {})
    seg["never_split"] = {"pk": round(ns[0], 3), "windowdiff": round(ns[1], 3)}
    seg["bilstm"] = {"pk": round(bilstm[0], 3), "windowdiff": round(bilstm[1], 3),
                     "threshold": thr}
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  saved -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--ckpt", default=CKPT, help="checkpoint to evaluate")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--save", default=None,
                    help="write/update a metrics JSON (e.g. reports/metrics.json)")
    args = ap.parse_args()

    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"{args.ckpt} not found - run: python src/segmentation/train.py")
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

    if args.save:
        save_metrics(args.save, (ns_pk, ns_wd), (m_pk, m_wd), thr)


if __name__ == "__main__":
    main()
