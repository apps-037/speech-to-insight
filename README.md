# Speech to Insight

CS6120 NLP - Final Project. "Speech → NLP Downstream."

A pipeline that turns meeting/lecture audio into organized, readable output:

```
Audio → Whisper → Transcript → Topic segmentation → Summarization → Evaluation
```

For a recording, the final output is the meeting split into topic sections, each
with a summary, plus an overall summary. On top of the pipeline, our key analysis
is **ASR-error propagation**: how much do Whisper's transcription mistakes degrade
downstream segmentation and summaries? We measure this by running the same models
on (a) clean human transcripts and (b) Whisper transcripts of the same meetings.

## Scope

Two downstream tasks: **topic segmentation** and **summarization**.

| Component                | Approach                                      | Trained by us?                |
| ------------------------ | --------------------------------------------- | ----------------------------- |
| Whisper (speech-to-text) | Pretrained, used as-is                        | No (it's the "Speech →" step) |
| Topic segmentation       | Per-sentence boundary classifier, built by us | Yes                           |
| Summarization            | Pretrained model **fine-tuned** on our data   | Yes (fine-tuning)             |

## Datasets

- **QMSum** - primary, for training and evaluation. Text only (no audio). Gives
  labels for both tasks: topic boundaries and summary pairs.
  https://github.com/Yale-LILY/QMSum
- **AMI** - audio, for running Whisper and the ASR-error analysis. It has both the
  audio and a clean human transcript for each meeting. We used 5 meetings (EN2001a,
  EN2001b, EN2001d, EN2001e, EN2003a), not the whole.
  https://huggingface.co/datasets/edinburghcstr/ami

## Repository layout

```
demo.py                       # one-command demo: runs the whole pipeline on predefind and user input

src/fetch_ami.py              # AMI: stream one meeting from HF, stitch clips into a .wav
src/transcribe.py             # Whisper (faster-whisper, base) audio → transcript .txt

src/segmentation/             # topic segmentation (BiLSTM boundary classifier)
  data.py  embed.py  model.py #   data prep · MiniLM sentence embeddings · BiLSTM
  metrics.py  baselines.py    #   Pk/WindowDiff · never-split & embedding-similarity baselines
  train.py  evaluate.py       #   training · evaluation
  segment_transcript.py       #   inference entry: transcript → topic sections

src/summarization/            # summarization (fine-tuned distilBART)
  data.py                     #   QMSum JSON → (input, target) pairs
  train.py                    #   fine-tune distilBART on QMSum
  evaluate.py                 #   ROUGE, pretrained vs fine-tuned
  summarize.py                #   inference entry: transcript → summary

src/pipeline.py               # end-to-end: transcript → segments → per-section + overall summary
src/error_analysis.py         # ASR error analysis: clean vs Whisper (WER + how far topic splits move); --meetings for a batch
src/plots.py                  # makes the results figures from reports/metrics.json

tests/test_summarize.py       # unit tests for the summarization package (chunking + data prep)
tests/test_segmentation.py    # unit tests for the segmentation package (labels, metrics, decoding)
data/audio/                   # AMI .wav files (gitignored)
data/transcripts/             # Whisper transcripts
data/reference/               # clean human reference transcripts (for error analysis)
data/qmsum/                   # cloned QMSum dataset (gitignored)
data/summaries/               # summary outputs (gitignored)
models/                       # trained checkpoints (gitignored - too large)
```

## Progress

**Speech-to-Text:**

- Python 3.12 environment; dependencies installed. (Python 3.14 breaks the ML libs.)
- `src/fetch_ami.py`: streams one AMI meeting and stitches it into a single `.wav`,
  avoiding the full 29 GB download. Quick-test and `--full` modes.
- `src/transcribe.py`: runs Whisper on an audio file, saves the transcript.
- Ran Whisper on 5 AMI meetings end to end (EN2001a, b, d, e and EN2003a), and kept
  each meeting's clean human transcript next to it for the error analysis.

**Topic Segmentation:**

- `src/segmentation/data.py`: QMSum `topic_list` spans → per-turn 0/1 boundary
  labels (boundaries are only ~1% of turns - QMSum topics are coarse).
- `src/segmentation/embed.py`: caches frozen MiniLM (all-MiniLM-L6-v2) sentence
  embeddings, so training/eval run in seconds on CPU.
- `src/segmentation/model.py`: a BiLSTM boundary classifier - the model we build and
  train ourselves (the frozen embeddings are only input features).
- `src/segmentation/train.py` + `evaluate.py`: weighted-BCE training with validation
  threshold tuning; Pk/WindowDiff on the QMSum test split. The trained model beats
  both baselines:

  | Method (QMSum test)  | Pk ↓  | WindowDiff ↓ |
  | -------------------- | ----- | ------------ |
  | never-split baseline | 0.355 | 0.357        |
  | embedding-similarity | 0.381 | 0.408        |
  | **BiLSTM (ours)**    | 0.344 | 0.350        |

- `src/segmentation/segment_transcript.py`: inference entry - splits a transcript into
  topic sections (windowed target-count decode).

**Summarization - fine-tuned:**

- `src/summarization/summarize.py`: turns a transcript into a summary. A meeting is
  longer than the model can read, so it summarizes the text in chunks and combines
  those into one summary. It uses the fine-tuned checkpoint if it's there, and
  prepends the same query it was trained with.
- `src/summarization/data.py`: builds the training pairs from QMSum. It strips the
  speaker names and puts a short query in front. Splits are 162 / 35 / 37
  (train / val / test). QMSum is cloned to `data/qmsum/` (gitignored).
- `src/summarization/train.py`: fine-tunes `distilbart-cnn-12-6` on those pairs
  (3 epochs, about 12 min on the Mac GPU). Saved to `models/distilbart-qmsum`, and also
  uploaded to the HF Hub since the 1.1 GB checkpoint is too big to commit
  (https://huggingface.co/appsaini602/distilbart-qmsum). The code downloads it from there
  when the local copy is absent.
- `src/summarization/evaluate.py`: ROUGE on the QMSum test split, before vs after
  fine-tuning:

  | Metric  | Pretrained | Fine-tuned | Change |
  | ------- | ---------- | ---------- | ------ |
  | ROUGE-1 | 0.2322     | 0.3809     | +64%   |
  | ROUGE-2 | 0.0475     | 0.1126     | +137%  |
  | ROUGE-L | 0.1454     | 0.2246     | +54%   |

  On the EN2001a Whisper transcript the fine-tuned model reads like real meeting
  notes, where the pretrained one gave choppy news-style text.

**Pipeline + ASR-error analysis:**

- `src/pipeline.py`: ties the two halves together. It takes a transcript, splits it into
  topic sections with our segmenter, summarizes each section with the fine-tuned model,
  and adds one overall summary. It uses the local fine-tuned checkpoint if it's there,
  otherwise it downloads the same fine-tuned model from the HF Hub
  (https://huggingface.co/appsaini602/distilbart-qmsum), so the summaries stay fine-tuned
  quality. We use it to turn the AMI Whisper transcripts into topic-wise notes (saved to
  `data/summaries/`).
- `src/error_analysis.py`: our main analysis. We run the same segmenter on two versions of
  each meeting, the clean human transcript and the Whisper one, and check how much the
  output changes. The clean AMI transcript has no punctuation and Whisper adds it, so
  comparing sentence by sentence isn't fair, so we cut both into fixed 25-word windows
  instead. Pass `--meetings` to run it over several meetings and average them.

  Across 5 meetings, Whisper got about 24% of the words wrong on average, but the topic
  boundaries only moved about 3%. So the segmentation holds up pretty well even when the
  transcript is noisy.

  | Meeting | WER   | boundary drift |
  | ------- | ----- | -------------- |
  | EN2001a | 0.262 | 0.041          |
  | EN2001b | 0.233 | 0.018          |
  | EN2001d | 0.263 | 0.024          |
  | EN2001e | 0.254 | 0.033          |
  | EN2003a | 0.186 | 0.031          |
  | avg     | 0.240 | 0.029          |

  The summary side of this (ROUGE between the clean and Whisper summaries) runs with
  `--summaries`.

## Setup

Use Python 3.11 or 3.12 (not 3.14 - it breaks the ML libraries).

```bash
python3.11 -m venv venv 
source venv/bin/activate
pip install --upgrade pip
pip install faster-whisper "datasets<3.0" soundfile numpy
# summarization:
pip install torch transformers datasets accelerate rouge-score sentencepiece
# segmentation + error analysis:
pip install nltk jiwer
```

### Quick demo

`demo.py` runs the whole pipeline (audio -> topic notes). Two ways to run it:

```bash
python demo.py                          # pre-defined input: a bundled sample meeting
python demo.py path/to/audio.wav        # your own audio (transcribed with Whisper first)
python demo.py path/to/transcript.txt   # your own transcript (skips Whisper)
```

With no arguments it transcribes the bundled meeting audio with Whisper, splits it into
topic sections, and summarizes each one. The summarizer uses the local fine-tuned
checkpoint if present, otherwise it downloads our fine-tuned copy from the HF Hub:
https://huggingface.co/appsaini602/distilbart-qmsum . The individual stages are below.

Run the speech-to-text pipeline:

```bash
python src/fetch_ami.py                         # quick 3-min sample
python src/fetch_ami.py EN2001a --full          # full meeting
python src/transcribe.py data/audio/EN2001a_sample.wav
```

Run segmentation, the full pipeline, and the ASR-error analysis:

```bash
# one-time: cache QMSum embeddings, then train + evaluate the segmenter
python src/segmentation/embed.py --split train      # also: val / test
python src/segmentation/train.py
python src/segmentation/evaluate.py

# end-to-end meeting notes for one meeting (saved to data/summaries/)
python src/pipeline.py data/transcripts/EN2001a.txt --num-sections 8 --out data/summaries/EN2001a.txt

# ASR-error propagation across several meetings
python src/error_analysis.py --meetings EN2001a EN2001b EN2001d EN2001e EN2003a --num-sections 8
```

Run the unit tests:

```bash
python tests/test_segmentation.py
python tests/test_summarize.py
```

## Notes

- Whisper on CPU is slow (~an hour for a 90-min meeting); use Colab's free GPU for
  real transcription and for fine-tuning.
- `datasets` 3.x wants `torchcodec`; pin `datasets<3.0` to decode audio via soundfile.
