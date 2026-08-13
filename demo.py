"""
Quick end-to-end demo: audio (or a transcript) goes in, topic notes come out.

Ways to run it:
    python demo.py                          # runs on the bundled sample meeting
    python demo.py path/to/audio.wav        # audio -> transcript -> notes
    python demo.py path/to/transcript.txt   # skips straight to notes

Audio files get transcribed with Whisper first; .txt files are used as-is. You
need the trained segmenter at models/segmenter.pt. The summarizer uses the local
fine-tuned checkpoint if it's there, otherwise it pulls our copy off the HF Hub.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))

SAMPLE_AUDIO = "data/audio/EN2003a.wav"              # the bundled sample meeting
SAMPLE_TRANSCRIPT = "data/transcripts/EN2003a.txt"   # used if the audio isn't there
SEG_CKPT = "models/segmenter.pt"
AUDIO_EXT = (".wav", ".mp3", ".m4a", ".flac", ".ogg")


def choose_input():
    if len(sys.argv) > 1:
        return sys.argv[1]                 # whatever the user passed
    if os.path.exists(SAMPLE_AUDIO):
        return SAMPLE_AUDIO                # default to the sample audio
    return SAMPLE_TRANSCRIPT              


def main():
    if not os.path.exists(SEG_CKPT):
        print(f"[demo] Segmenter checkpoint not found at {SEG_CKPT}. Train it first (see README).")
        sys.exit(1)

    inp = choose_input()
    if not inp or not os.path.exists(inp):
        print(f"[demo] Input not found: {inp}")
        print("[demo] Pass an audio file or a transcript:  python demo.py <path>")
        sys.exit(1)

    if inp.lower().endswith(AUDIO_EXT):
        import transcribe
        print(f"[demo] Transcribing audio with Whisper (the slow step): {inp}")
        transcribe.transcribe(inp, verbose=False)  
        base = os.path.splitext(os.path.basename(inp))[0]
        transcript = f"data/transcripts/{base}.txt"
    else:
        transcript = inp

    import pipeline
    print(f"\n[demo] Segmentation + summarization on: {transcript}\n")
    notes = pipeline.run(transcript, query=pipeline.summarize.DEFAULT_QUERY)   # sections auto-scale with meeting length

    base = os.path.splitext(os.path.basename(transcript))[0]
    out_path = f"data/summaries/{base}.txt"
    os.makedirs("data/summaries", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(notes + "\n")
    print(f"\n[demo] Saved notes to {out_path}")
    print("[demo] Done.")


if __name__ == "__main__":
    main()
