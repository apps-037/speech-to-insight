"""
Fetch a single AMI meeting from HuggingFace and reconstruct it into one audio file.

AMI on HuggingFace (edinburghcstr/ami) is pre-chunked into thousands of tiny
utterance clips. For our project we want a *full meeting* to transcribe and later
segment, so this script:
  1. streams only the rows for ONE meeting (no 29 GB download)
  2. sorts the clips by their start time
  3. concatenates them into a single .wav file in data/audio/

Usage:
    python src/fetch_ami.py                # uses default meeting EN2001a
    python src/fetch_ami.py ES2004a        # pick a different meeting id

Requirements (add to your venv):
    pip install datasets soundfile numpy
"""

import sys
import os
import numpy as np
import soundfile as sf
from datasets import load_dataset

# AMI has 36 meetings. A few valid ids: EN2001a, ES2004a, IS1000a, TS3003a
DEFAULT_MEETING = "EN2001a"
SAMPLING_RATE = 16000  # AMI audio is 16 kHz
QUICK_LIMIT = 60            # clips to grab in quick mode
SCAN_SAFETY_LIMIT = 130000  # stop scanning after this many rows

def fetch_meeting(meeting_id, full=False):
    limit = None if full else QUICK_LIMIT
    mode = "FULL meeting" if full else f"quick test (first {QUICK_LIMIT} clips)"
    print(f"Streaming AMI for meeting '{meeting_id}' - {mode}")
    print("(streaming mode - only downloads clips as it reads them)\n")

    ds = load_dataset("edinburghcstr/ami", "ihm", split="train", streaming=True)

    clips = []
    scanned = 0
    last_match = 0
    for row in ds:
        scanned += 1
        if scanned % 500 == 0:
            print(f"  scanned {scanned} rows, collected {len(clips)} clips...", flush=True)

        if row["meeting_id"] == meeting_id:
            last_match = scanned 
            clips.append({
                "begin": row["begin_time"],
                "audio": row["audio"]["array"],
                "text": row["text"],
            })
            if limit is not None and len(clips) >= limit:
                print(f"  reached quick-test limit of {limit} clips, stopping scan.")
                break
        
        if len(clips) > 0 and (scanned - last_match) > 1000:
            print(f"  past the meeting block, stopping at {len(clips)} clips.")
            break

        if scanned >= SCAN_SAFETY_LIMIT:
            print("  reached scan safety limit, stopping.")
            break

    if not clips:
        print(f"No clips found for meeting '{meeting_id}'.")
        print("Try one of: EN2001a, ES2004a, IS1000a, TS3003a")
        sys.exit(1)

    print(f"Found {len(clips)} clips. Sorting and stitching...")

    # Sort clips into chronological order using their start time
    clips.sort(key=lambda c: c["begin"])

    # Concatenate all clip audio into one long array
    full_audio = np.concatenate([c["audio"] for c in clips])

    # Save the reconstructed meeting audio
    os.makedirs("data/audio", exist_ok=True)
    suffix = "" if full else "_sample"
    out_audio = f"data/audio/{meeting_id}{suffix}.wav"
    sf.write(out_audio, full_audio, SAMPLING_RATE)

    duration_min = len(full_audio) / SAMPLING_RATE / 60
    print(f"Saved {out_audio}  (~{duration_min:.1f} minutes of audio)")

    # Also save the reference transcript (AMI's own human transcript).
    # This is your "clean" transcript for the ASR-error comparison later.
    os.makedirs("data/reference", exist_ok=True)
    out_ref = f"data/reference/{meeting_id}.txt"
    with open(out_ref, "w") as f:
        f.write(" ".join(c["text"] for c in clips))
    print(f"Saved reference transcript to {out_ref}")

    print("\nNext step:")
    print(f"    python src/transcribe.py data/audio/{meeting_id}.wav")


if __name__ == "__main__":
    args = sys.argv[1:]
    full = "--full" in args
    args = [a for a in args if a != "--full"]
    meeting = args[0] if args else DEFAULT_MEETING
    fetch_meeting(meeting, full=full)
