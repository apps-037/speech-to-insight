from faster_whisper import WhisperModel
import sys
import os

# Use "base" or "small" for quick testing; "medium" is more accurate but slower
MODEL_SIZE = "base"

def transcribe(audio_path, verbose=True, progress_min=3):
    # compute_type="int8" keeps it fast and light on CPU
    model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")

    segments, info = model.transcribe(audio_path, beam_size=5)

    print(f"Detected language: {info.language} (probability {info.language_probability:.2f})")

    full_text = []
    step = max(1, progress_min) * 60   # in quiet mode, a progress line every few minutes
    next_mark = step
    for segment in segments:
        if verbose:   # per-segment lines when run as a standalone script
            print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
        else:
            while segment.end >= next_mark:
                print(f"  transcribed ~{next_mark // 60} min of audio...", flush=True)
                next_mark += step
        full_text.append(segment.text)

    # Save transcript
    base = os.path.splitext(os.path.basename(audio_path))[0]
    os.makedirs("data/transcripts", exist_ok=True)
    out_path = f"data/transcripts/{base}.txt"
    with open(out_path, "w") as f:
        f.write(" ".join(full_text).strip())
    print(f"\nSaved transcript to {out_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/transcribe.py <path_to_audio>")
        sys.exit(1)
    transcribe(sys.argv[1])