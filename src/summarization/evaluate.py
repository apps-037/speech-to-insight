"""
Evaluate a summarization model on the QMSum test split with ROUGE.

Runs the model on QMSum inputs (truncated to the model's max input, standard
QMSum + BART practice) and reports ROUGE-1/2/L F-measure against the reference
summaries. Use it to compare the pretrained baseline vs. the fine-tuned model.

    python src/summarization/evaluate.py                    # pretrained baseline
    python src/summarization/evaluate.py --model models/ft  # fine-tuned checkpoint
    python src/summarization/evaluate.py --limit 20         # quick subset
"""
import argparse

import torch
from rouge_score import rouge_scorer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.utils import logging as hf_logging

from data import load_qmsum_pairs
from summarize import summarize_chunk  # reuse single-chunk summarizer

hf_logging.set_verbosity_error()
DEFAULT_MODEL = "sshleifer/distilbart-cnn-12-6"


def pick_device(device):
    if device:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def evaluate(model_name, split="test", limit=None, device=None,
             max_summary=160, min_summary=30):
    device = pick_device(device)
    print(f"Loading '{model_name}' on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    model.eval()

    pairs = load_qmsum_pairs(split)
    if limit:
        pairs = pairs[:limit]

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    agg = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    for i, p in enumerate(pairs, 1):
        pred = summarize_chunk(p["input"], model, tokenizer, device,
                               max_input=1024, max_summary=max_summary,
                               min_summary=min_summary)
        sc = scorer.score(p["target"], pred)
        for k in agg:
            agg[k] += sc[k].fmeasure
        if i == 1:
            print("\n--- sample (first example) ---")
            print("PRED:", pred[:300])
            print("REF :", p["target"][:300])
            print("-------------------------------")
        if i % 10 == 0:
            print(f"  {i}/{len(pairs)}")

    n = len(pairs)
    results = {k: agg[k] / n for k in agg}
    print(f"\nmodel: {model_name}   split: {split}   n={n}")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")
    return results


def main():
    ap = argparse.ArgumentParser(description="ROUGE evaluation on QMSum.")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-summary", type=int, default=160)
    ap.add_argument("--min-summary", type=int, default=30)
    args = ap.parse_args()
    evaluate(args.model, args.split, args.limit, args.device,
             args.max_summary, args.min_summary)


if __name__ == "__main__":
    main()
