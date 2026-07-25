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
src/qmsum_prep.py          # QMSum JSON → (input, target) summary pairs
src/train_summarizer.py    # [planned] fine-tune distilBART/T5 on QMSum
src/evaluate_summary.py    # ROUGE on QMSum, pretrained vs fine-tuned
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

**Done (summarization - fine-tuned):**

- `src/summarize.py`: transcript `.txt` → summary `.txt`. Map-reduce over
  token-bounded chunks (chunk → summarize → recursively reduce to one summary), so
  arbitrarily long meetings fit the model. The chunker is a pure, unit-tested
  function (`tests/test_summarize.py`) that a topic-segmentation boundary function
  can replace later with no rewrite. Auto-uses the fine-tuned checkpoint when present
  and prepends the same query used in training.
- `src/qmsum_prep.py`: builds (input, target) pairs from QMSum general queries -
  query-prefixed, speaker labels stripped to match the Whisper blob. 162 / 35 / 37
  train / val / test pairs. QMSum cloned to `data/qmsum/` (gitignored).
- `src/train_summarizer.py`: real `Seq2SeqTrainer` fine-tuning loop. Fine-tuned
  `distilbart-cnn-12-6` for 3 epochs (~12 min on the Mac's MPS GPU), saved to
  `models/distilbart-qmsum`.
- `src/evaluate_summary.py`: ROUGE-1/2/L on the QMSum test split, pretrained vs
  fine-tuned:

  | Metric  | Baseline (pretrained) | Fine-tuned | Change |
  | ------- | --------------------- | ---------- | ------ |
  | ROUGE-1 | 0.2322                | 0.3809     | +64%   |
  | ROUGE-2 | 0.0475                | 0.1126     | +137%  |
  | ROUGE-L | 0.1454                | 0.2246     | +54%   |

  Confirmed qualitatively on the real EN2001a Whisper transcript: the fine-tuned
  model gives a coherent, structured meeting summary where the pretrained baseline
  gave disjointed news-style text.

**Next:**

- Swap the fixed chunker for the teammate's topic-segmentation boundaries
  (summary per topic + overall) once it lands.
- ASR-error analysis: run the summarizer on clean reference vs Whisper transcripts
  of the same meetings and compare (the project's headline analysis).
- Optional: enlarge training data with QMSum specific-query pairs; move the
  fine-tune to Colab if we scale up.

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
