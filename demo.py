"""
demo.py - a standalone demo of the Speech-to-Insight pipeline (audio -> topic notes).

Two ways to run it:

  1. Pre-defined input (no arguments) - runs on a bundled sample meeting:
       python demo.py

  2. Your own input:
       python demo.py path/to/audio.wav        # audio -> transcript -> topic notes
       python demo.py path/to/transcript.txt   # transcript -> topic notes

Audio inputs (.wav, .mp3, ...) are transcribed with Whisper first; .txt inputs are
treated as a transcript and go straight to segmentation + summarization.

Needs the trained segmenter at models/segmenter.pt. The summarizer uses the local
fine-tuned checkpoint if present, otherwise it downloads our fine-tuned copy from
the HF Hub.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))

SAMPLE_AUDIO = "data/audio/EN2003a.wav"              # pre-defined input (a full meeting)
SAMPLE_TRANSCRIPT = "data/transcripts/EN2003a.txt"   # fallback if the audio isn't present
SEG_CKPT = "models/segmenter.pt"
AUDIO_EXT = (".wav", ".mp3", ".m4a", ".flac", ".ogg")


def choose_input():
    if len(sys.argv) > 1:
        return sys.argv[1]                 # user-provided input
    if os.path.exists(SAMPLE_AUDIO):
        return SAMPLE_AUDIO                # pre-defined input (audio)
    return SAMPLE_TRANSCRIPT               # pre-defined fallback (transcript)


def main():
    if not os.path.exists(SEG_CKPT):
        print(f"[demo] Segmenter checkpoint not found at {SEG_CKPT}. Train it first (see README).")
        sys.exit(1)

    inp = choose_input()
    if not inp or not os.path.exists(inp):
        print(f"[demo] Input not found: {inp}")
        print("[demo] Pass an audio file or a transcript:  python demo.py <path>")
        sys.exit(1)

    # audio -> transcribe with Whisper first; text -> use it as the transcript
    if inp.lower().endswith(AUDIO_EXT):
        import transcribe
        print(f"[demo] Transcribing audio with Whisper (the slow step): {inp}\n")
        transcribe.transcribe(inp)
        base = os.path.splitext(os.path.basename(inp))[0]
        transcript = f"data/transcripts/{base}.txt"
    else:
        transcript = inp

    import pipeline
    print(f"\n[demo] Segmentation + summarization on: {transcript}\n")
    pipeline.run(transcript, num_sections=6, query=pipeline.summarize.DEFAULT_QUERY)
    print("\n[demo] Done.")


if __name__ == "__main__":
    main()
