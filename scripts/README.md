# Scripts — Quick Reference

Each script is a clean, standalone Python file extracted from the
`notebooks/moonshine-be-v1.ipynb` experiment notebook.
Run them **in order** for a full training pipeline.

---

## Pipeline overview

```
prepare_data.py   →   train_tokenizer.py   →   train.py
                                                   ↓
                                            transcribe.py
                                            evaluate.py
                                            upload_kaggle.py
```

---

## 1. `prepare_data.py` — Data preparation

Reads the raw [Frovolts-Random](https://www.kaggle.com/datasets/sanjidh090/frovolts-random/data)
Kaggle dataset, filters clips by duration (1–30 s) and minimum transcript
length, copies audio to the working directory, and writes three TSV split
files (`train.tsv`, `dev.tsv`, `test.tsv`).

```bash
# Kaggle (default paths)
python scripts/prepare_data.py

# Local override
DATA_ROOT=/data/frovolts-random OUTPUT_DIR=/data/lipi-ghor \
    python scripts/prepare_data.py
```

**Outputs** (inside `OUTPUT_DIR`, default `/kaggle/working/lipi-ghor`):
- `audio/`  — filtered WAV files
- `train.tsv`, `dev.tsv`, `test.tsv`  — `relative_audio_path\ttranscript`

---

## 2. `train_tokenizer.py` — SentencePiece tokenizer training

Cleans the transcripts (Bengali Unicode range, punctuation), trains a
32 768-piece unigram SentencePiece model, and runs round-trip smoke tests.

```bash
python scripts/train_tokenizer.py

# Point at a different metadata CSV
METADATA_CSV=/kaggle/working/metadata.csv \
    python scripts/train_tokenizer.py
```

**Output**: `OUTPUT_DIR/tokenizer/bn_unigram.model`

---

## 3. `train.py` — Fine-tuning

Loads `UsefulSensors/moonshine-base`, swaps the decoder embedding table
and LM head to the Bengali vocab size, freezes the encoder, and trains
for 15 epochs with cosine LR decay.  
Best checkpoint (lowest validation loss) is saved automatically.

```bash
python scripts/train.py

# Key overrides
EPOCHS=20 BATCH_SIZE=16 LR=1e-4 python scripts/train.py
```

**Output**: `SAVE_DIR/best/`  (default `/kaggle/working/checkpoints/best`)
- `model_state.pt`
- `config.json`
- `bn_unigram.model`

---

## 4. `transcribe.py` — Inference

Transcribes a single short clip or a long recording (sliding-window).

```bash
# Short clip
python scripts/transcribe.py --audio clip.wav

# Long recording (auto-chunked)
python scripts/transcribe.py --audio long.wav --mode longform

# Custom checkpoint
python scripts/transcribe.py \
    --audio clip.wav \
    --checkpoint /kaggle/working/checkpoints/best \
    --sp-model   /kaggle/working/lipi-ghor/tokenizer/bn_unigram.model
```

---

## 5. `evaluate.py` — WER / CER evaluation

Computes Word Error Rate (WER) and Character Error Rate (CER) using
[jiwer](https://github.com/jitsi/jiwer).

```bash
# Single file vs. a reference text
python scripts/evaluate.py \
    --audio  path/to/audio.wav \
    --ref    path/to/reference.txt

# Full test split
python scripts/evaluate.py \
    --tsv        /kaggle/working/lipi-ghor/test.tsv \
    --audio-base /kaggle/working/lipi-ghor

# Quick sanity check (first 50 samples)
python scripts/evaluate.py \
    --tsv  test.tsv --audio-base /kaggle/working/lipi-ghor \
    --max-samples 50
```

---

## 6. `upload_kaggle.py` — Publish checkpoint to Kaggle

Packages the checkpoint directory and uploads it as a Kaggle dataset.

```bash
# Set credentials (or use ~/.kaggle/kaggle.json)
export KAGGLE_USERNAME=sanjidh090
export KAGGLE_KEY=<your-api-key>

# First upload (creates a new dataset)
python scripts/upload_kaggle.py --mode create

# Push an updated version later
python scripts/upload_kaggle.py --mode version --message "epoch 20 run"
```

---

## Dependencies

```
pip install torch transformers sentencepiece soundfile pandas jiwer kaggle
# optional for non-16kHz audio:
pip install librosa
```
