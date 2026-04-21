# Moonshine ASR Bangla

Bangla ASR experimentation built around **UsefulSensors/moonshine-base** and a custom Bangla tokenizer + finetuning pipeline.

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
│   └── README.md                  # dataset link placeholders
└── notebooks/
    └── moonshine-be-v1.ipynb      # main experiment notebook
```

## Main Notebook

- `notebooks/moonshine-be-v1.ipynb`:
  Complete workflow for preparation, tokenizer training, model adaptation, finetuning, and evaluation.

## Dataset Links (Placeholders)

Please fill these links later.

| Purpose | Link |
|---|---|
| Primary training dataset | TODO |
| Validation dataset | TODO |
| Test dataset | TODO |
| Auxiliary/extra dataset | TODO |

More detailed placeholders are also in `data/README.md`.

## Next Steps

- add finalized dataset links and short dataset notes
- continue finetuning with more data + compute
- compare Moonshine variants / decoding settings
- track benchmark metrics (WER/CER) in a consistent log format
