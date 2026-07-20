from faster_whisper import WhisperModel
import sys
import os

# Use "base" or "small" for quick testing; "medium" is more accurate but slower
MODEL_SIZE = "base"

def transcribe(audio_path):
    # compute_type="int8" keeps it fast and light on CPU
    model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")

    segments, info = model.transcribe(audio_path, beam_size=5)

    print(f"Detected language: {info.language} (probability {info.language_probability:.2f})")

    full_text = []
    for segment in segments:
        line = f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}"
        print(line)
        full_text.append(segment.text)

    # Save transcript
    base = os.path.splitext(os.path.basename(audio_path))[0]
    out_path = f"data/transcripts/{base}.txt"
    with open(out_path, "w") as f:
        f.write(" ".join(full_text).strip())
    print(f"\nSaved transcript to {out_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/transcribe.py <path_to_audio>")
        sys.exit(1)
    transcribe(sys.argv[1])