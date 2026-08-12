"""
End-to-end meeting pipeline: transcript -> topic sections -> per-section
summaries + one overall summary.

Glue connecting the two halves of the project:
  - segmentation (src/segmentation/segment_transcript.py) cuts the transcript
    into topic sections;
  - summarization (src/summarization/summarize.py) summarizes each section, then
    reduces the section summaries into one overall summary.

The summarizer is used as-is: `summarize.DEFAULT_MODEL` resolves to the
fine-tuned checkpoint (models/distilbart-qmsum) if it exists, else the
pretrained baseline. So this runs today on the pretrained model and upgrades for
free once the fine-tuned checkpoint is in place - nothing here changes.

    python src/pipeline.py data/transcripts/EN2001a.txt --num-sections 6

Heads-up: summarization is the heavy step (many beam-search generations on CPU/
MPS). Keep --num-sections small for a first run.

Imports: both halves are run-as-script packages using flat sibling imports, so
we put each package dir on sys.path (segmentation first, so its `data`/`model`/
`embed` win over summarization's same-named modules) rather than importing them
as packages.
"""
import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "summarization"))
sys.path.insert(0, os.path.join(_HERE, "segmentation"))   # inserted last -> first on path

from segment_transcript import segment_text, load_model as load_segmenter, Embedder, pick_device  # noqa: E402
import summarize  # noqa: E402

SEG_CKPT = "models/segmenter.pt"


def run(transcript, num_sections=None, target_len=40, seg_ckpt=SEG_CKPT,
        sum_model=None, device=None, max_summary=120, min_summary=25, query=None):
    with open(transcript, encoding="utf-8") as f:
        text = f.read().strip()
    device = pick_device(device)
    sum_model = sum_model or summarize.DEFAULT_MODEL

    print(f"[1/2] Segmenting on {device} ...")
    seg_model = load_segmenter(seg_ckpt, device)
    embedder = Embedder(device)
    sections, sents, bounds = segment_text(
        text, seg_model, embedder, device,
        target_len=target_len, num_sections=num_sections)
    print(f"      {len(sents)} sentences -> {len(sections)} sections")

    print(f"[2/2] Summarizing {len(sections)} sections with '{sum_model}' ...")
    sm, tok = summarize.load_model(sum_model, device)
    section_summaries = []
    for i, sec in enumerate(sections, 1):
        s = summarize.summarize_document(sec, sm, tok, device,
                                         max_summary=max_summary,
                                         min_summary=min_summary, query=query)
        section_summaries.append(s)
        print(f"      section {i}/{len(sections)} done")

    overall = summarize.summarize_document(" ".join(section_summaries), sm, tok,
                                           device, max_summary=max_summary,
                                           min_summary=min_summary, query=query)

    lines = [f"MEETING NOTES: {os.path.basename(transcript)}", "-" * 64,
             "", "OVERALL SUMMARY", "  " + overall]
    for i, start in enumerate(bounds):
        end = bounds[i + 1] - 1 if i + 1 < len(bounds) else len(sents) - 1
        summ = section_summaries[i]
        lines += ["", f"▸ Section {i + 1}  (sentences {start}-{end})", f"  {summ}"]
    notes = "\n".join(lines)
    print("\n" + notes)
    return notes


def main():
    ap = argparse.ArgumentParser(description="Transcript -> topic sections -> summaries.")
    ap.add_argument("transcript", help="path to a transcript .txt file")
    ap.add_argument("--num-sections", type=int, default=None,
                    help="exact number of sections (else auto from --target-len)")
    ap.add_argument("--target-len", type=int, default=40)
    ap.add_argument("--seg-ckpt", default=SEG_CKPT)
    ap.add_argument("--model", default=None,
                    help="summarizer model/checkpoint (default: fine-tuned if present, else pretrained)")
    ap.add_argument("--device", default=None, help="cpu | mps | cuda (default: auto)")
    ap.add_argument("--max-summary", type=int, default=120)
    ap.add_argument("--min-summary", type=int, default=25)
    ap.add_argument("--query", default=summarize.DEFAULT_QUERY,
                    help="query prefix for the summarizer; pass '' to disable")
    ap.add_argument("--out", default=None, help="also write the notes to this path")
    args = ap.parse_args()

    notes = run(args.transcript, args.num_sections, args.target_len, args.seg_ckpt,
                args.model, args.device, args.max_summary, args.min_summary,
                args.query or None)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(notes + "\n")
        print(f"\nSaved notes to {args.out}")


if __name__ == "__main__":
    main()
