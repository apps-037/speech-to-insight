"""
Fine-tune a seq2seq summarizer on QMSum.

Trains with a real forward/loss/backprop loop (HuggingFace Seq2SeqTrainer) on the
QMSum general-summary pairs from data.py, then saves the fine-tuned model.
Evaluate before/after with src/summarization/evaluate.py.

    # quick loop check (tiny, fast):
    python src/summarization/train.py --model t5-small --limit 8 --epochs 1 \
        --max-input 512 --output-dir models/debug

    # real run:
    python src/summarization/train.py         # distilBART, 3 epochs -> models/distilbart-qmsum
"""
import argparse
import os

from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from transformers.utils import logging as hf_logging

from data import load_qmsum_pairs

hf_logging.set_verbosity_error()
DEFAULT_MODEL = "sshleifer/distilbart-cnn-12-6"


def build_dataset(split, tokenizer, max_input, max_target, limit=None):
    pairs = load_qmsum_pairs(split)
    if limit:
        pairs = pairs[:limit]
    ds = Dataset.from_list(pairs)

    def tokenize(batch):
        model_inputs = tokenizer(batch["input"], max_length=max_input, truncation=True)
        labels = tokenizer(text_target=batch["target"], max_length=max_target, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return ds.map(tokenize, batched=True, remove_columns=ds.column_names)


def main():
    ap = argparse.ArgumentParser(description="Fine-tune a summarizer on QMSum.")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--output-dir", default="models/distilbart-qmsum")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--max-input", type=int, default=1024)
    ap.add_argument("--max-target", type=int, default=160)
    ap.add_argument("--limit", type=int, default=None, help="cap training examples (debug)")
    args = ap.parse_args()

    print(f"Loading '{args.model}'...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)

    train_ds = build_dataset("train", tokenizer, args.max_input, args.max_target, args.limit)
    print(f"Training examples: {len(train_ds)}")

    collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    train_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
        fp16=False,  # MPS/CPU: keep fp32
        bf16=False,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        data_collator=collator,
    )

    trainer.train()

    os.makedirs(args.output_dir, exist_ok=True)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nSaved fine-tuned model to {args.output_dir}")


if __name__ == "__main__":
    main()
