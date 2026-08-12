"""
demo.py - a standalone demo of the Speech-to-Insight pipeline.

Runs the whole thing on a short sample meeting transcript:
    transcript -> topic segmentation -> a summary per section + one overall summary

    python demo.py                      # uses a bundled sample transcript
    python demo.py path/to/transcript.txt

Notes for a fresh checkout:
  - Needs the trained segmenter at models/segmenter.pt. If it's missing, train it with
    the three commands the script prints.
  - Uses the fine-tuned summarizer at models/distilbart-qmsum if it's there; otherwise
    it falls back to the pretrained DistilBART (downloaded from HuggingFace), so the demo
    still runs, just with rougher summaries.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))

# first one that exists is used
SAMPLE_TRANSCRIPTS = [
    "data/transcripts/EN2003a.txt",
    "data/transcripts/EN2001a.txt",
]
SEG_CKPT = "models/segmenter.pt"


def pick_sample():
    for path in SAMPLE_TRANSCRIPTS:
        if os.path.exists(path):
            return path
    return None


def main():
    if not os.path.exists(SEG_CKPT):
        print(f"[demo] Segmenter checkpoint not found at {SEG_CKPT}.")
        print("[demo] Train it first:")
        print("[demo]   python src/segmentation/embed.py --split train")
        print("[demo]   python src/segmentation/embed.py --split val")
        print("[demo]   python src/segmentation/train.py")
        sys.exit(1)

    transcript = sys.argv[1] if len(sys.argv) > 1 else pick_sample()
    if not transcript or not os.path.exists(transcript):
        print("[demo] No sample transcript found.")
        print("[demo] Pass one:  python demo.py <path/to/transcript.txt>")
        sys.exit(1)

    import pipeline  # imported here so the checkpoint check runs first

    print(f"[demo] Running the pipeline on: {transcript}")
    print("[demo] (segmentation -> summarization; the summary step can take a minute)\n")
    pipeline.run(transcript, num_sections=6, query=pipeline.summarize.DEFAULT_QUERY)
    print("\n[demo] Done.")


if __name__ == "__main__":
    main()
