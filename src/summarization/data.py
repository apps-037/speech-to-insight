"""Build (input, target) summarization pairs from QMSum.

input  = "<query> <meeting transcript>" with speaker labels stripped.
target = the human whole-meeting summary (QMSum general_query_list answer).

Usage: python src/summarization/data.py [--dump]
"""
import argparse
import json
import os
import statistics

QMSUM_DIR = "data/qmsum/data/ALL/jsonl"
SPLIT_FILES = {"train": "train.jsonl", "val": "val.jsonl", "test": "test.jsonl"}


def strip_speakers(meeting_transcripts):
    """Join turn contents into one string, dropping speaker labels."""
    return " ".join(turn["content"].strip() for turn in meeting_transcripts)


def load_qmsum_pairs(split, data_dir=QMSUM_DIR, prepend_query=True):
    """Return a list of {input, target, query} dicts for a QMSum split."""
    path = os.path.join(data_dir, SPLIT_FILES[split])
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            transcript = strip_speakers(rec["meeting_transcripts"])
            for gq in rec.get("general_query_list", []):
                query = gq["query"].strip()
                answer = gq["answer"].strip()
                if not answer:
                    continue
                inp = f"{query} {transcript}" if prepend_query else transcript
                pairs.append({"input": inp, "target": answer, "query": query})
    return pairs


def main():
    """Print per-split pair counts and length stats; with --dump, write jsonl."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=QMSUM_DIR)
    ap.add_argument("--dump", action="store_true",
                    help="write processed jsonl to data/qmsum_processed/")
    args = ap.parse_args()

    out_dir = "data/qmsum_processed"
    total = 0
    for split in SPLIT_FILES:
        pairs = load_qmsum_pairs(split, args.data_dir)
        total += len(pairs)
        in_words = [len(p["input"].split()) for p in pairs]
        tg_words = [len(p["target"].split()) for p in pairs]
        print(f"{split:5s}: {len(pairs):4d} pairs | "
              f"input words avg {int(statistics.mean(in_words)):5d} max {max(in_words):6d} | "
              f"target words avg {int(statistics.mean(tg_words)):3d} max {max(tg_words):4d}")
        if args.dump:
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, f"{split}.jsonl"), "w", encoding="utf-8") as f:
                for p in pairs:
                    f.write(json.dumps(p) + "\n")
    print(f"total: {total} pairs")
    if args.dump:
        print(f"wrote processed pairs to {out_dir}/")


if __name__ == "__main__":
    main()
