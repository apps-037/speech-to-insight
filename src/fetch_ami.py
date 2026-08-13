"""
Grab one AMI meeting off HuggingFace and stitch its clips back into a single wav.

The HF version (edinburghcstr/ami) is split into thousands of tiny utterance
clips, but we want a whole meeting to transcribe and segment. So we stream just
the rows for one meeting (no 29 GB download), sort them by start time, and
concatenate them into one file under data/audio/.

    python src/fetch_ami.py                # default meeting EN2001a
    python src/fetch_ami.py ES2004a        # some other meeting id

Needs: pip install datasets soundfile numpy
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

    # put the clips back in time order
    clips.sort(key=lambda c: c["begin"])

    # glue all the clip audio into one long array
    full_audio = np.concatenate([c["audio"] for c in clips])

    # write out the reconstructed meeting audio
    os.makedirs("data/audio", exist_ok=True)
    suffix = "" if full else "_sample"
    out_audio = f"data/audio/{meeting_id}{suffix}.wav"
    sf.write(out_audio, full_audio, SAMPLING_RATE)

    duration_min = len(full_audio) / SAMPLING_RATE / 60
    print(f"Saved {out_audio}  (~{duration_min:.1f} minutes of audio)")

    # also dump AMI's own human transcript. we use it as the "clean" text
    # to compare against the Whisper output in the ASR-error analysis.
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
