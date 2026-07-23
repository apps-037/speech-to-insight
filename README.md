# Speech to Insight

CS6120 NLP - Final Project. Based on idea #23, "Speech → NLP Downstream."

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

Two downstream tasks: **topic segmentation** and **summarization**. (Intent
detection was an option in the project idea; we deliberately skipped it.)

Per the professor's ruling, we do **not** train from scratch. Fine-tuning a model
for our use case is acceptable; the only thing not allowed is running an existing
model on a test set with no training at all.

| Component                | Approach                                      | Trained by us?                |
| ------------------------ | --------------------------------------------- | ----------------------------- |
| Whisper (speech-to-text) | Pretrained, used as-is                        | No (it's the "Speech →" step) |
| Topic segmentation       | Per-sentence boundary classifier, built by us | Yes                           |
| Summarization            | Pretrained model **fine-tuned** on our data   | Yes (fine-tuning)             |

## Datasets

- **QMSum** - primary, for training and evaluation. Text only (no audio). Gives
  labels for both tasks: topic boundaries and summary pairs.
  https://github.com/Yale-LILY/QMSum
- **AMI** - audio, for running Whisper and the ASR-error analysis. Has both audio
  and clean reference transcripts. We use only 3–5 meetings, not the full 29 GB.
  https://huggingface.co/datasets/edinburghcstr/ami

## Repository layout

```
src/fetch_ami.py           # AMI: stream one meeting from HF, stitch clips into a .wav
src/transcribe.py          # Whisper (faster-whisper, base) audio → transcript .txt
src/summarize.py           # transcript .txt → summary .txt (the deliverable)
src/qmsum_prep.py          # [planned] QMSum JSON → (text, summary) training pairs
src/train_summarizer.py    # [planned] fine-tune distilBART/T5 on QMSum
src/evaluate_summary.py    # [planned] ROUGE, pretrained vs fine-tuned
tests/test_summarize.py    # unit tests for the chunker
notebooks/finetune.ipynb   # [planned] Colab GPU training run
data/audio/                # AMI .wav files (gitignored)
data/transcripts/          # Whisper transcripts
data/reference/            # clean human reference transcripts (for error analysis)
data/qmsum/                # cloned QMSum dataset (gitignored)
data/summaries/            # summary outputs (gitignored)
models/                    # [planned] fine-tuned checkpoint (gitignored - too large)
```

## Progress

**Done (speech-to-text half):**

- Python 3.12 environment; dependencies installed. (Python 3.14 breaks the ML libs.)
- `src/fetch_ami.py`: streams one AMI meeting and stitches it into a single `.wav`,
  avoiding the full 29 GB download. Quick-test and `--full` modes.
- `src/transcribe.py`: runs Whisper on an audio file, saves the transcript.
- Transcribed a real AMI meeting (EN2001a) end to end; clean reference transcript
  stored alongside it for the error analysis.

**Done (summarization - pretrained baseline):**

- `src/summarize.py`: transcript `.txt` → summary `.txt`. Map-reduce over
  token-bounded chunks (chunk → summarize → recursively reduce to one summary), so
  arbitrarily long meetings fit the model. The chunker is a pure, unit-tested
  function (`tests/test_summarize.py`) that a topic-segmentation boundary function
  can replace later with no rewrite.
- Verified end to end on the real EN2001a Whisper transcript (19k tokens → 26 chunks
  → one summary) using pretrained `sshleifer/distilbart-cnn-12-6`. This is the
  **"before" baseline** - output is rough because the model was trained on news, not
  meetings; fine-tuning on QMSum is what improves it.
- QMSum dataset cloned to `data/qmsum/` (splits at `data/ALL/jsonl/`).

**Next (summarization - fine-tuning):**

- `src/qmsum_prep.py`: QMSum meeting text → general summary, speaker tags stripped so
  training matches the unlabeled blob seen at inference.
- `src/train_summarizer.py`: fine-tune (debug on `t5-small` locally, real run on
  Colab's free T4 GPU - Mac CPU too slow for the full fine-tune).
- `src/evaluate_summary.py`: ROUGE on the QMSum test split, pretrained baseline vs
  fine-tuned, to demonstrate the fine-tuning effect.

## Setup

Use Python 3.11 or 3.12 (not 3.14 - it breaks the ML libraries).

```bash
python3.11 -m venv venv        # or: python3.12 -m venv venv312
source venv/bin/activate
pip install --upgrade pip
pip install faster-whisper "datasets<3.0" soundfile numpy
# summarization (added as that component lands):
pip install torch transformers evaluate rouge-score sentencepiece accelerate nltk
```

Run the speech-to-text pipeline:

```bash
python src/fetch_ami.py                         # quick 3-min sample
python src/fetch_ami.py EN2001a --full          # full meeting
python src/transcribe.py data/audio/EN2001a_sample.wav
```

## Notes

- Whisper on CPU is slow (~an hour for a 90-min meeting); use Colab's free GPU for
  real transcription and for fine-tuning.
- `datasets` 3.x wants `torchcodec`; pin `datasets<3.0` to decode audio via soundfile.
