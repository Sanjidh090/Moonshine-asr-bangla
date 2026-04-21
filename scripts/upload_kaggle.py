"""
upload_kaggle.py
────────────────
Packages the fine-tuned Bengali Moonshine checkpoint and uploads it as
a new (or updated) Kaggle dataset.

Prerequisites:
  pip install kaggle
  Either set KAGGLE_USERNAME / KAGGLE_KEY env vars,
  or place ~/.kaggle/kaggle.json with {"username":…,"key":…}.

Usage:
    # Create a new dataset (first upload)
    python scripts/upload_kaggle.py --mode create

    # Push a new version to an existing dataset
    python scripts/upload_kaggle.py --mode version --message "epoch 15 checkpoint"

Env-var overrides (all optional):
    UPLOAD_DIR       – directory to package   (default: /kaggle/working/checkpoints/best)
    DATASET_TITLE    – human-readable title
    DATASET_SLUG     – URL slug  (letters, numbers, hyphens only)
    KAGGLE_USERNAME  – Kaggle account username
    KAGGLE_KEY       – Kaggle API key
"""

import argparse
import json
import os
import subprocess
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────────
UPLOAD_DIR    = Path(os.getenv("UPLOAD_DIR",    "/kaggle/working/checkpoints/best"))
DATASET_TITLE = os.getenv("DATASET_TITLE",      "Moonshine Bengali ASR v1")
DATASET_SLUG  = os.getenv("DATASET_SLUG",       "moonshine-bengali-asr-v1")
KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME",  "")
KAGGLE_KEY      = os.getenv("KAGGLE_KEY",       "")
# ─────────────────────────────────────────────────────────────────────────────


def set_credentials():
    if KAGGLE_USERNAME and KAGGLE_KEY:
        os.environ["KAGGLE_USERNAME"] = KAGGLE_USERNAME
        os.environ["KAGGLE_KEY"]      = KAGGLE_KEY
        print(f"Using credentials for: {KAGGLE_USERNAME}")
    else:
        # Fall back to ~/.kaggle/kaggle.json
        kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
        if not kaggle_json.exists():
            raise RuntimeError(
                "No Kaggle credentials found. Set KAGGLE_USERNAME / KAGGLE_KEY "
                "env vars, or create ~/.kaggle/kaggle.json."
            )
        with open(kaggle_json) as f:
            creds = json.load(f)
        os.environ["KAGGLE_USERNAME"] = creds["username"]
        os.environ["KAGGLE_KEY"]      = creds["key"]
        print(f"Using credentials from ~/.kaggle/kaggle.json  ({creds['username']})")


def write_metadata(upload_dir: Path):
    username = os.environ["KAGGLE_USERNAME"]
    metadata = {
        "title": DATASET_TITLE,
        "id": f"{username}/{DATASET_SLUG}",
        "licenses": [{"name": "CC0-1.0"}],
        "subtitle": "Fine-tuned Bengali Moonshine ASR checkpoint + SentencePiece tokenizer",
    }
    meta_path = upload_dir / "dataset-metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Metadata written to: {meta_path}")
    return meta_path


def run_cmd(cmd: str):
    print(f"\n$ {cmd}")
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True
    )
    if result.stdout:
        print(result.stdout)
    if result.returncode != 0:
        print("STDERR:", result.stderr)
        raise RuntimeError(f"Command failed (exit {result.returncode}): {cmd}")


def main():
    parser = argparse.ArgumentParser(description="Upload checkpoint to Kaggle")
    parser.add_argument(
        "--mode", choices=["create", "version"], default="create",
        help="'create' for a new dataset, 'version' to update an existing one",
    )
    parser.add_argument(
        "--message", default="New checkpoint",
        help="Version message (only used with --mode version)",
    )
    parser.add_argument(
        "--upload-dir", default=str(UPLOAD_DIR),
        help=f"Directory to upload (default: {UPLOAD_DIR})",
    )
    args = parser.parse_args()

    upload_dir = Path(args.upload_dir)
    if not upload_dir.exists():
        raise FileNotFoundError(
            f"Upload directory not found: {upload_dir}\n"
            "Run train.py first to generate the checkpoint."
        )

    set_credentials()
    write_metadata(upload_dir)

    if args.mode == "create":
        run_cmd(f"kaggle datasets create -p {upload_dir} --dir-mode zip")
        username = os.environ["KAGGLE_USERNAME"]
        print(f"\n✅ Dataset created: https://www.kaggle.com/{username}/{DATASET_SLUG}")
    else:
        run_cmd(
            f'kaggle datasets version -p {upload_dir} --dir-mode zip -m "{args.message}"'
        )
        username = os.environ["KAGGLE_USERNAME"]
        print(f"\n✅ New version uploaded: https://www.kaggle.com/{username}/{DATASET_SLUG}")


if __name__ == "__main__":
    main()
