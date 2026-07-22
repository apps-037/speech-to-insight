# Speech to Insight

Summarization and topic segmentation of lecture and meeting audio.

CS6120 Natural Language Processing — Final Project (based on idea #23, "Speech → NLP Downstream").

## Overview

Long recordings like lectures and meetings are hard to skim or review, and the
text you get from transcribing them is messy and unstructured. This project builds
a pipeline that:

1. Transcribes meeting audio with Whisper (speech → text).
2. Splits the transcript into topic-based sections (topic segmentation).
3. Summarizes each section and the whole meeting (summarization).

It also analyzes how transcription errors from Whisper affect the quality of the
downstream segmentation and summaries (the ASR-error-propagation analysis).

## Pipeline

```
Audio (AMI) -> Whisper -> Transcript -> Topic segmentation -> Summarization -> Evaluation
```

## Models

- Whisper — pretrained, used as-is for the speech-to-text step.
- Topic segmentation — trained by us from scratch (per-sentence boundary
  classifier). Planned.
- Summarization — a pretrained model fine-tuned on our data. Planned.

Note: the professor confirmed that fine-tuning a pretrained model satisfies the
"train your own model" requirement; only using a pretrained model as-is on a test
set (with no training) is not acceptable.

## Datasets

- QMSum (primary, for training and evaluation) — provides topic-segmentation
  annotations and query/summary pairs, all as text.
  https://github.com/Yale-LILY/QMSum
- AMI (audio, for running Whisper and the ASR-error analysis) — real meeting
  recordings with reference transcripts. Used via the HuggingFace version.
  https://huggingface.co/datasets/edinburghcstr/ami

## Project structure

```
speech-to-insight/
├── data/
│   ├── audio/          # meeting audio (.wav), gitignored
│   ├── transcripts/    # Whisper-generated transcripts, gitignored
│   └── reference/      # AMI human reference transcripts (clean), gitignored
├── src/
│   ├── fetch_ami.py    # download one AMI meeting and stitch clips into one .wav
│   └── transcribe.py   # run Whisper on an audio file to produce a transcript
├── notes/
├── requirements.txt
└── README.md
```

## Setup

Use Python 3.12 (newer versions such as 3.14 currently break some of the ML
libraries).

```bash
python3.12 -m venv venv312
source venv312/bin/activate
pip install --upgrade pip
pip install faster-whisper "datasets<3.0" soundfile numpy
```

Remember to activate the environment (`source venv312/bin/activate`) in each new
terminal session.

## Usage

### 1. Fetch AMI audio

Quick test (first 60 clips, ~3 minutes, saves `EN2001a_sample.wav`):

```bash
python src/fetch_ami.py
```

Full meeting (saves `EN2001a.wav`):

```bash
python src/fetch_ami.py EN2001a --full
```

Other meeting ids can be passed the same way, e.g. `ES2004a`, `IS1000a`.

### 2. Transcribe with Whisper

```bash
python src/transcribe.py data/audio/EN2001a_sample.wav
```

The transcript is written to `data/transcripts/`. The reference (clean) transcript
from AMI is saved separately to `data/reference/` for the later error analysis.

## Progress

Done so far (Week 1):

- Set up the repo, Python 3.12 environment, and installed dependencies.
- Wrote `fetch_ami.py` to stream a single AMI meeting from HuggingFace and
  reconstruct it into one audio file (avoids downloading the full 29 GB corpus).
- Wrote `transcribe.py` to run Whisper and save transcripts.
- Successfully transcribed a real AMI meeting sample end to end.

Next:

- Load QMSum and inspect its topic-boundary and summary labels.
- Build and train the topic segmentation model.
- Fine-tune the summarization model.
- Run the ASR-error-propagation analysis (clean vs. Whisper transcripts).

## Notes

- Whisper on CPU is slow; a full ~90-minute meeting can take an hour or more.
  Google Colab's free GPU is a faster option for transcription.
- AMI audio is chunked into short utterances and normalized (uppercase, no
  punctuation); `fetch_ami.py` reconstructs full meetings by sorting on start time.

## AI usage

To be maintained per the course AI Tool Usage Policy: document which tools were
used, for what, and which files were affected, distinguishing agent-generated
infrastructure from hand-written core components.