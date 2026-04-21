# Moonshine ASR Bangla

Bangla ASR experimentation built around **UsefulSensors/moonshine-base** and a custom Bangla tokenizer + finetuning pipeline.
# Until now,notebook is the only thing I have touched,,,copilot is for decoration, work ongoing
## Owner

- Sanjid Hasan (`sanjidh090`)

## Current Status

This repository currently contains the first end-to-end experimental notebook (`moonshine-be-v1`, where **be = Bangla Experiment**) and documents what has already been done:

- dataset preparation and train/dev/test split generation
- Bangla text cleaning and SentencePiece tokenizer training
- adaptation of Moonshine Base for Bangla vocabulary
- finetuning/evaluation experiments on Bangla speech data
- sample inference (single clip + long-form chunking)

## What Worked So Far

- Initial Bangla pipeline runs successfully from data prep to inference.
- Model can produce understandable Bangla transcriptions in multiple examples.
- Training loss decreases during finetuning, showing learning behavior.

## Current Limitations

From current experiments, the first `moonshine-base` adaptation still has notable issues:

- limited generalization beyond seen patterns
- inconsistent pattern capture in harder/long-form audio
- high error rates in several evaluation samples

Likely next improvement direction: **larger and more diverse dataset + more compute for longer/better training**.

## Repository Structure

```text
Moonshine-asr-bangla/
├── README.md
├── data/
│   └── README.md                   # dataset links and notes
├── notebooks/
│   └── moonshine-be-v1.ipynb       # original experiment notebook
└── scripts/
    ├── README.md                   # quick-reference guide for all scripts
    ├── prepare_data.py             # filter & split the raw dataset into TSVs
    ├── train_tokenizer.py          # train Bengali SentencePiece tokenizer
    ├── train.py                    # adapt + fine-tune Moonshine for Bengali
    ├── transcribe.py               # single-clip and long-form inference
    ├── evaluate.py                 # WER / CER evaluation
    └── upload_kaggle.py            # publish checkpoint to Kaggle
```

## Standalone Scripts

The `scripts/` directory contains clean, self-contained Python scripts
extracted from the notebook — easier to read, run individually, and reuse:

| Script | Purpose |
|---|---|
| `prepare_data.py` | Filter clips by duration, write train/dev/test TSV splits |
| `train_tokenizer.py` | Clean Bengali text, train 32 k SentencePiece model |
| `train.py` | Swap vocab, freeze encoder, fine-tune decoder |
| `transcribe.py` | Greedy inference — single clip or sliding-window long-form |
| `evaluate.py` | WER / CER against reference text or a full TSV split |
| `upload_kaggle.py` | Package checkpoint and upload to Kaggle |

See `scripts/README.md` for full usage and CLI options.

## Main Notebook

- `notebooks/moonshine-be-v1.ipynb`:
  Original end-to-end experiment notebook (reference / exploration).

## Dataset Links

| Purpose | Link |
|---|---|
| Primary training dataset (Frovolts Random) | [sanjidh090/frovolts-random](https://www.kaggle.com/datasets/sanjidh090/frovolts-random/data) |
| Validation dataset | TODO |
| Test dataset | TODO |
| Auxiliary/extra dataset | TODO |

> The primary dataset may be extended further in future iterations.

More detailed notes are also in `data/README.md`.

## Next Steps

- add finalized dataset links and short dataset notes
- continue finetuning with more data + compute
- compare Moonshine variants / decoding settings
- track benchmark metrics (WER/CER) in a consistent log format
