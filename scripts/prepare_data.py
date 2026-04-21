"""
prepare_data.py
───────────────
Reads the raw Frovolts-Random dataset from Kaggle, filters clips by
duration and character count, then writes train / dev / test TSV splits
(relative audio path + transcript) to OUTPUT_DIR.

Usage (Kaggle kernel):
    python scripts/prepare_data.py

Usage (local — override paths with env vars or edit CONFIG below):
    DATA_ROOT=/path/to/frovolts-random OUTPUT_DIR=/path/to/out \
        python scripts/prepare_data.py
"""

import csv
import os
import random
import shutil
import soundfile as sf
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────────
INPUT_ROOT = Path(
    os.getenv("DATA_ROOT",
              "/kaggle/input/datasets/sanjidh090/frovolts-random")
)
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/kaggle/working/lipi-ghor"))

SPLIT      = {"train": 0.90, "dev": 0.05, "test": 0.05}
MIN_DUR    = 1.0    # seconds
MAX_DUR    = 30.0   # seconds
MIN_CHARS  = 3      # minimum transcript characters
SEED       = 42
# ─────────────────────────────────────────────────────────────────────────────

random.seed(SEED)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
audio_out = OUTPUT_DIR / "audio"
audio_out.mkdir(exist_ok=True)


def get_duration(wav_path: Path) -> float:
    """Return duration in seconds for a WAV file without decoding fully."""
    try:
        info = sf.info(str(wav_path))
        return info.duration
    except Exception:
        return 0.0


def main():
    meta_path = INPUT_ROOT / "wavs_asr_chunks" / "metadata.csv"
    wav_root  = INPUT_ROOT / "wavs_asr_chunks" / "wavs"

    print(f"Reading metadata from: {meta_path}")
    rows: list[dict] = []
    with open(meta_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    print(f"  Total rows in metadata: {len(rows):,}")

    # Detect the column names (flexible)
    sample_row = rows[0]
    file_col = next(
        (c for c in sample_row if "file" in c.lower() or "wav" in c.lower()), None
    )
    text_col = next(
        (c for c in sample_row if "text" in c.lower() or "transcript" in c.lower()), None
    )
    if file_col is None or text_col is None:
        raise ValueError(
            f"Cannot find file/text columns in: {list(sample_row.keys())}"
        )
    print(f"  Using columns: file='{file_col}', text='{text_col}'")

    # Filter and copy audio
    accepted: list[tuple[str, str]] = []
    skipped = 0
    for row in rows:
        fname      = Path(row[file_col]).name
        transcript = row[text_col].strip()
        src        = wav_root / fname

        if len(transcript) < MIN_CHARS:
            skipped += 1
            continue
        if not src.exists():
            skipped += 1
            continue

        dur = get_duration(src)
        if dur < MIN_DUR or dur > MAX_DUR:
            skipped += 1
            continue

        dst = audio_out / fname
        if not dst.exists():
            shutil.copy2(src, dst)

        accepted.append((f"audio/{fname}", transcript))

    print(f"  Accepted: {len(accepted):,}  |  Skipped: {skipped:,}")

    # Shuffle & split
    random.shuffle(accepted)
    n        = len(accepted)
    n_train  = int(n * SPLIT["train"])
    n_dev    = int(n * SPLIT["dev"])

    splits = {
        "train": accepted[:n_train],
        "dev":   accepted[n_train : n_train + n_dev],
        "test":  accepted[n_train + n_dev :],
    }

    for split_name, items in splits.items():
        out_path = OUTPUT_DIR / f"{split_name}.tsv"
        with open(out_path, "w", encoding="utf-8") as f:
            for rel, text in items:
                f.write(f"{rel}\t{text}\n")
        print(f"  Wrote {len(items):,} lines → {out_path}")

    print("\nDone. Dataset splits are ready in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
