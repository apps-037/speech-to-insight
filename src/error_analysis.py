"""
ASR-error propagation analysis: how much do Whisper's transcription mistakes
change the downstream segmentation and summaries?

Runs the same models on two versions of ONE meeting - the clean human reference
transcript vs. the Whisper transcript - and compares the outputs:

  input degradation   WER(clean, whisper)   how bad the ASR is (needs `jiwer`)
  segmentation drift   segment both into the same N sections, compare boundary
                       positions normalized to [0,1] (the two texts differ in
                       length). 0 = identical splits; higher = splits moved.
  summary drift        (--summaries) summarize both, ROUGE(whisper-summary vs
                       clean-summary) = how much the summary content changed.
                       Heavy; best run with the fine-tuned checkpoint. Needs
                       `rouge-score`.

    python src/error_analysis.py                        # EN2001a, seg + WER
    python src/error_analysis.py --num-sections 8 --summaries   # + summary drift

Imports: same flat-path setup as pipeline.py (segmentation dir first).
"""
import argparse
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "summarization"))
sys.path.insert(0, os.path.join(_HERE, "segmentation"))   # first on path

from segment_transcript import (segment_text, load_model as load_segmenter,  # noqa: E402
                                 Embedder, pick_device)

SEG_CKPT = "models/segmenter.pt"
CLEAN = "data/reference/EN2001a.txt"
WHISPER = "data/transcripts/EN2001a.txt"


def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read().strip()


def _norm(s):
    return " ".join(re.sub(r"[^\w\s]", " ", s.lower()).split())


def split_words(text, per=25):
    """Split into fixed windows of `per` words - punctuation-agnostic units so
    the clean (unpunctuated) and Whisper transcripts are compared on equal
    footing (the AMI reference has no punctuation; Whisper adds it)."""
    words = text.split()
    return [" ".join(words[i:i + per]) for i in range(0, len(words), per)] or [text]


def try_wer(clean, whisper):
    """WER(clean, whisper) if jiwer is installed, else None."""
    try:
        import jiwer
    except ImportError:
        return None
    return jiwer.wer(_norm(clean), _norm(whisper))


def norm_positions(bounds, n):
    return [b / n for b in bounds if b != 0]   # fractional positions; drop 0


def segmentation_drift(clean_b, clean_n, whisper_b, whisper_n):
    """Mean nearest-neighbour distance between the two sets of boundary
    positions (normalized to [0,1]). 0 = splits at the same relative places."""
    c, w = norm_positions(clean_b, clean_n), norm_positions(whisper_b, whisper_n)
    if not c or not w:
        return None
    return sum(min(abs(x - y) for y in w) for x in c) / len(c)


def summary_drift(sections_clean, sections_whisper, model_name, device):
    import summarize
    from rouge_score import rouge_scorer
    model_name = model_name or summarize.DEFAULT_MODEL
    print(f"  loading summarizer '{model_name}' ...")
    sm, tok = summarize.load_model(model_name, device)

    def summarize_all(secs):
        parts = [summarize.summarize_document(s, sm, tok, device) for s in secs]
        return summarize.summarize_document(" ".join(parts), sm, tok, device)

    clean_sum = summarize_all(sections_clean)
    whisper_sum = summarize_all(sections_whisper)
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(clean_sum, whisper_sum)   # target=clean, prediction=whisper
    return clean_sum, whisper_sum, scores


def analyze_pair(clean_text, whisper_text, seg_model, embedder, device,
                 num_sections, words_per_unit):
    """WER + segmentation drift for one clean/Whisper pair."""
    wer = try_wer(clean_text, whisper_text)
    cunits = split_words(clean_text, words_per_unit)
    wunits = split_words(whisper_text, words_per_unit)
    _, csent, cb = segment_text(clean_text, seg_model, embedder, device,
                                num_sections=num_sections, units=cunits)
    _, wsent, wb = segment_text(whisper_text, seg_model, embedder, device,
                                num_sections=num_sections, units=wunits)
    drift = segmentation_drift(cb, len(csent), wb, len(wsent))
    return wer, drift


def save_asr(rows, mean_wer, mean_drift, path="reports/metrics.json"):
    """Write per-meeting + average ASR numbers into metrics.json (for plots.py)."""
    import json
    data = {}
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
    data["asr"] = {
        "per_meeting": [{"meeting": m, "wer": round(w, 3), "drift": round(d, 3)}
                        for m, w, d in rows],
        "mean_wer": round(mean_wer, 3),
        "mean_drift": round(mean_drift, 3),
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nsaved -> {path}")


def run_batch(meetings, args, device):
    """Run the analysis over several meetings and report a table + averages."""
    seg_model = load_segmenter(args.seg_ckpt, device)
    embedder = Embedder(device)
    rows = []
    for mid in meetings:
        clean_text = read(f"data/reference/{mid}.txt")
        whisper_text = read(f"data/transcripts/{mid}.txt")
        wer, drift = analyze_pair(clean_text, whisper_text, seg_model, embedder,
                                  device, args.num_sections, args.words_per_unit)
        rows.append((mid, wer, drift))
        print(f"  {mid}:  WER={wer:.3f}  drift={drift:.3f}")

    mean_wer = sum(r[1] for r in rows) / len(rows)
    mean_drift = sum(r[2] for r in rows) / len(rows)

    print("\n=== ASR-error propagation across meetings ===")
    print(f"{'meeting':<10}{'WER':>10}{'drift':>10}")
    for mid, wer, drift in rows:
        print(f"{mid:<10}{wer:>10.3f}{drift:>10.3f}")
    print("-" * 30)
    print(f"{'AVERAGE':<10}{mean_wer:>10.3f}{mean_drift:>10.3f}")
    print(f"\n-> across {len(rows)} meetings: Whisper changed ~{mean_wer * 100:.0f}% "
          f"of words, but topic boundaries moved only ~{mean_drift * 100:.0f}%")
    save_asr(rows, mean_wer, mean_drift)


def main():
    ap = argparse.ArgumentParser(description="ASR-error propagation analysis.")
    ap.add_argument("--clean", default=CLEAN)
    ap.add_argument("--whisper", default=WHISPER)
    ap.add_argument("--num-sections", type=int, default=8)
    ap.add_argument("--words-per-unit", type=int, default=25,
                    help="word-window size for the fixed-unit segmentation comparison")
    ap.add_argument("--seg-ckpt", default=SEG_CKPT)
    ap.add_argument("--device", default=None)
    ap.add_argument("--summaries", action="store_true",
                    help="also run the (heavy) summarization comparison")
    ap.add_argument("--model", default=None, help="summarizer model/checkpoint")
    ap.add_argument("--meetings", nargs="+", default=None,
                    help="meeting IDs to batch over (expects data/reference/<id>.txt "
                         "and data/transcripts/<id>.txt)")
    args = ap.parse_args()

    device = pick_device(args.device)

    if args.meetings:
        run_batch(args.meetings, args, device)
        return

    clean_text, whisper_text = read(args.clean), read(args.whisper)

    print("=== ASR-error propagation: clean vs Whisper ===")
    print(f"clean:   {args.clean}\nwhisper: {args.whisper}\n")

    # 1. input degradation
    wer = try_wer(clean_text, whisper_text)
    if wer is None:
        print("WER: install jiwer to compute  ->  pip install jiwer")
    else:
        print(f"INPUT  WER(clean, whisper) = {wer:.3f}  ({wer * 100:.1f}% of words changed)")

    # 2. segmentation drift - fixed word-windows so the clean (unpunctuated) and
    #    Whisper (punctuated) transcripts are segmented on equal footing.
    print("\nsegmenting both transcripts (fixed word-windows) ...")
    seg_model = load_segmenter(args.seg_ckpt, device)
    embedder = Embedder(device)
    cunits = split_words(clean_text, args.words_per_unit)
    wunits = split_words(whisper_text, args.words_per_unit)
    cs, csent, cb = segment_text(clean_text, seg_model, embedder, device,
                                 num_sections=args.num_sections, units=cunits)
    ws, wsent, wb = segment_text(whisper_text, seg_model, embedder, device,
                                 num_sections=args.num_sections, units=wunits)
    drift = segmentation_drift(cb, len(csent), wb, len(wsent))
    print(f"SEG    {args.num_sections} sections each  (units = {args.words_per_unit}-word windows)")
    print(f"       clean:   {len(csent)} windows, boundaries {cb}")
    print(f"       whisper: {len(wsent)} windows, boundaries {wb}")
    print(f"       boundary drift = {drift:.3f}  "
          f"(0 = identical splits; ~{drift * 100:.1f}% of the meeting length)")

    print("\n--- what this means ------------------------------")
    if wer is not None:
        print(f"  Whisper changed ~{wer * 100:.0f}% of the words   (WER {wer:.3f})")
    print(f"  but the topic boundaries moved only ~{drift * 100:.0f}%")
    print("  -> segmentation is fairly robust to ASR errors")

    # 3. summary drift (optional, heavy)
    if args.summaries:
        print("\nSUMMARY drift (heavy step) ...")
        clean_sum, whisper_sum, scores = summary_drift(cs, ws, args.model, device)
        print("       ROUGE(whisper-summary vs clean-summary):")
        for k, v in scores.items():
            print(f"         {k}: F={v.fmeasure:.3f}")
        print(f"\n       CLEAN summary:\n         {clean_sum}")
        print(f"\n       WHISPER summary:\n         {whisper_sum}")


if __name__ == "__main__":
    main()
