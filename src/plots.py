"""
Draw the evaluation figures (segmentation, ROUGE, ASR) from reports/metrics.json into
reports/figures/.

Usage:
    python src/plots.py
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METRICS = "reports/metrics.json"
FIGDIR = "reports/figures"

plt.rcParams["font.family"] = "Arial"
plt.rcParams["axes.edgecolor"] = "#B7C0C8"

TEAL, GRAY, GREEN, INK = "#1C7293", "#9AA5B1", "#3B8C5A", "#3E4C59"


def _labelled_bars(ax, x, heights, width, color, label, fmt):
    rects = ax.bar(x, heights, width, color=color, label=label,
                   edgecolor="white", linewidth=0.6)
    for r in rects:
        ax.annotate(fmt.format(r.get_height()),
                    (r.get_x() + r.get_width() / 2, r.get_height()),
                    ha="center", va="bottom", fontsize=8.5, color=INK,
                    xytext=(0, 2), textcoords="offset points")


def _finish(ax, title, ylab, top):
    ax.set_ylabel(ylab, fontsize=10, color=INK)
    ax.set_title(title, fontsize=12.5, fontweight="bold", color="#1F2933", pad=10)
    ax.set_ylim(0, top)
    ax.legend(frameon=False, fontsize=9.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=0, labelsize=10, colors=INK)


def seg_figure(d):
    methods = ["never-split", "embedding-sim", "BiLSTM (ours)"]
    keys = ["never_split", "embedding_similarity", "bilstm"]
    pk = [d[k]["pk"] for k in keys]
    wd = [d[k]["windowdiff"] for k in keys]
    x = np.arange(len(methods))
    w = 0.38
    fig, ax = plt.subplots(figsize=(6.4, 3.7))
    _labelled_bars(ax, x - w / 2, pk, w, TEAL, "Pk", "{:.2f}")
    _labelled_bars(ax, x + w / 2, wd, w, GRAY, "WindowDiff", "{:.2f}")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    _finish(ax, "Topic segmentation on QMSum test", "score  (lower is better)",
            max(max(pk), max(wd)) * 1.28)
    fig.tight_layout()
    return fig


def rouge_figure(d):
    labels = ["ROUGE-1", "ROUGE-2", "ROUGE-L"]
    ks = ["rouge1", "rouge2", "rougeL"]
    pre = [d["pretrained"][k] for k in ks]
    fin = [d["finetuned"][k] for k in ks]
    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(6.4, 3.7))
    _labelled_bars(ax, x - w / 2, pre, w, GRAY, "pretrained", "{:.2f}")
    _labelled_bars(ax, x + w / 2, fin, w, GREEN, "fine-tuned", "{:.2f}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    _finish(ax, "Summarization ROUGE on QMSum test", "F1  (higher is better)",
            max(fin) * 1.3)
    fig.tight_layout()
    return fig


def asr_figure(d):
    rows = d["per_meeting"]
    meetings = [r["meeting"] for r in rows] + ["AVG"]
    wer = [r["wer"] for r in rows] + [d["mean_wer"]]
    drift = [r["drift"] for r in rows] + [d["mean_drift"]]
    x = np.arange(len(meetings))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    _labelled_bars(ax, x - w / 2, wer, w, TEAL, "WER (input error)", "{:.2f}")
    _labelled_bars(ax, x + w / 2, drift, w, GREEN, "boundary drift (output change)", "{:.2f}")
    ax.set_xticks(x)
    ax.set_xticklabels(meetings, fontsize=9.5)
    _finish(ax, "ASR-error propagation across meetings", "rate  (lower = less change)",
            max(wer) * 1.3)
    fig.tight_layout()
    return fig


def main():
    with open(METRICS) as f:
        m = json.load(f)
    os.makedirs(FIGDIR, exist_ok=True)
    written = []
    seg_figure(m["segmentation"]).savefig(f"{FIGDIR}/segmentation.png", dpi=150)
    written.append("segmentation.png")
    rouge_figure(m["rouge"]).savefig(f"{FIGDIR}/rouge.png", dpi=150)
    written.append("rouge.png")
    if "asr" in m:
        asr_figure(m["asr"]).savefig(f"{FIGDIR}/asr.png", dpi=150)
        written.append("asr.png")
    print("wrote " + ", ".join(f"{FIGDIR}/{w}" for w in written))


if __name__ == "__main__":
    main()
