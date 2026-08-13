"""
Turn QMSum meetings into per-turn topic-boundary labels (0/1) for the segmenter.

Usage:
    python src/segmentation/data.py --split val --show 0
"""
import argparse
import json
import os

DEFAULT_QMSUM_DIR = "data/qmsum/data/ALL/jsonl"
SPLIT_FILES = {
    "train": ["train.jsonl"],
    "val":   ["val.jsonl", "valid.jsonl"],   # QMSum names it val; guard both
    "test":  ["test.jsonl"],
}


def _split_path(qmsum_dir, split):
    for name in SPLIT_FILES[split]:
        p = os.path.join(qmsum_dir, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"No file for split '{split}' in {qmsum_dir} "
        f"(looked for {SPLIT_FILES[split]}). "
        f"Clone QMSum: git clone https://github.com/Yale-LILY/QMSum data/qmsum")


def load_split(split, qmsum_dir=DEFAULT_QMSUM_DIR):
    """Return a list of raw QMSum meeting dicts for the given split."""
    with open(_split_path(qmsum_dir, split), encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def meeting_to_labels(meeting):
    """Meeting dict -> (list[str] turn texts, list[int] boundary labels)."""
    turns = meeting["meeting_transcripts"]
    n = len(turns)
    starts = {0}                       # first turn always opens a segment
    for topic in meeting.get("topic_list", []):
        for span in topic.get("relevant_text_span", []):
            i = int(span[0])
            if 0 <= i < n:
                starts.add(i)
    labels = [1 if i in starts else 0 for i in range(n)]
    texts = [t["content"] for t in turns]
    return texts, labels


def iter_labeled(split, qmsum_dir=DEFAULT_QMSUM_DIR, min_turns=5):
    """Yield (texts, labels) for each meeting with at least `min_turns` turns."""
    for meeting in load_split(split, qmsum_dir):
        texts, labels = meeting_to_labels(meeting)
        if len(texts) >= min_turns:
            yield texts, labels


def _show(meeting):
    texts, labels = meeting_to_labels(meeting)
    n_bound = sum(labels)
    print(f"{len(texts)} turns, {n_bound} boundaries "
          f"({n_bound / len(texts):.1%})\n")
    for i, (t, b) in enumerate(zip(texts, labels)):
        mark = ">>> NEW TOPIC" if b else "            "
        snippet = t.strip().replace("\n", " ")[:90]
        print(f"{mark} [{i:3}] {snippet}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qmsum-dir", default=DEFAULT_QMSUM_DIR)
    ap.add_argument("--split", default="val", choices=list(SPLIT_FILES))
    ap.add_argument("--show", type=int, default=0,
                    help="index of the meeting to print with boundaries marked")
    args = ap.parse_args()

    meetings = load_split(args.split, args.qmsum_dir)
    print(f"{args.split}: {len(meetings)} meetings\n")
    _show(meetings[args.show])


if __name__ == "__main__":
    main()
