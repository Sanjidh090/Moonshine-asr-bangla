"""
train_tokenizer.py
──────────────────
Trains a Bengali SentencePiece (unigram) tokenizer from the text
transcripts found in the metadata CSV produced by prepare_data.py.

The resulting model file is saved to OUTPUT_DIR/tokenizer/bn_unigram.model
and is consumed by train.py and transcribe.py.

Usage:
    python scripts/train_tokenizer.py

Override paths with env vars:
    METADATA_CSV=/path/to/metadata.csv OUTPUT_DIR=/path/to/lipi-ghor \
        python scripts/train_tokenizer.py
"""

import os
import re
import sentencepiece as spm
import pandas as pd
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────────
# The metadata CSV can be either the raw Kaggle one or the HuggingFace one.
METADATA_CSV = Path(
    os.getenv("METADATA_CSV", "/kaggle/working/metadata.csv")
)
OUTPUT_DIR   = Path(os.getenv("OUTPUT_DIR", "/kaggle/working/lipi-ghor"))

VOCAB_SIZE       = 32768
CHARACTER_COVERAGE = 0.9999
DUPLICATE_CORPUS   = 2      # light duplication for better coverage
# ─────────────────────────────────────────────────────────────────────────────

TOK_DIR = OUTPUT_DIR / "tokenizer"
TOK_DIR.mkdir(parents=True, exist_ok=True)

BN_RANGE = (0x0980, 0x09FF)
PUNCT    = "।॥?!,;:—-"


def is_mostly_bengali(text: str, threshold: float = 0.3) -> bool:
    chars = [c for c in text if c.isalpha()]
    if not chars:
        return False
    bn = sum(1 for c in chars if BN_RANGE[0] <= ord(c) <= BN_RANGE[1])
    return (bn / len(chars)) >= threshold


def clean(text: str) -> str:
    text = str(text).strip()
    text = re.sub(
        r"[^\u0980-\u09FFa-zA-Z0-9\s" + re.escape(PUNCT) + r"]",
        "",
        text,
    )
    return re.sub(r"\s+", " ", text).strip()


def main():
    print(f"Loading metadata from: {METADATA_CSV}")
    df = pd.read_csv(METADATA_CSV)

    # Auto-detect text column
    text_col = next(
        (c for c in df.columns if "text" in c.lower() or "transcript" in c.lower()),
        None,
    )
    if text_col is None:
        raise ValueError(f"Cannot find text column in: {list(df.columns)}")
    print(f"  Using text column: '{text_col}'")

    # Collect clean Bengali lines
    lines: list[str] = []
    for raw in df[text_col]:
        cleaned = clean(raw)
        if cleaned and is_mostly_bengali(cleaned) and len(cleaned.split()) > 2:
            lines.append(cleaned)

    print(f"  Clean lines collected: {len(lines):,}")
    lines = lines * DUPLICATE_CORPUS
    print(f"  After duplication ×{DUPLICATE_CORPUS}: {len(lines):,}")

    corpus_path = TOK_DIR / "corpus.txt"
    corpus_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Corpus written to: {corpus_path}")

    # Train SentencePiece
    print(f"\nTraining SentencePiece tokenizer (vocab_size={VOCAB_SIZE}) ...")
    spm.SentencePieceTrainer.train(
        input=str(corpus_path),
        model_prefix=str(TOK_DIR / "bn_unigram"),
        vocab_size=VOCAB_SIZE,
        model_type="unigram",
        character_coverage=CHARACTER_COVERAGE,
        input_sentence_size=2_000_000,
        shuffle_input_sentence=True,
        hard_vocab_limit=False,
        max_sentencepiece_length=16,
        pad_id=0,  unk_id=1,  bos_id=2,  eos_id=3,
        pad_piece="<pad>", unk_piece="<unk>",
        bos_piece="<s>",   eos_piece="</s>",
        user_defined_symbols=list(PUNCT),
        split_digits=False,
        normalization_rule_name="nmt_nfkc",
        byte_fallback=True,
    )
    model_path = TOK_DIR / "bn_unigram.model"
    print(f"Tokenizer saved to: {model_path}")

    # Smoke test
    sp = spm.SentencePieceProcessor()
    sp.load(str(model_path))
    print(f"\nVocab size: {sp.get_piece_size()}")

    test_sentences = [
        "আমার সোনার বাংলা",
        "আজকের আবহাওয়া কেমন?",
        "আমি আজকে বাজারে যাব",
        "মানসিক রোগ সম্পর্কে সচেতন হওয়া জরুরি",
    ]
    print("\nRound-trip tests:")
    all_ok = True
    for sent in test_sentences:
        pieces  = sp.encode_as_pieces(sent)
        decoded = sp.decode_pieces(pieces)
        ok      = decoded == sent
        status  = "✓" if ok else "✗"
        print(f"  {status}  '{sent}'  →  {pieces}")
        if not ok:
            all_ok = False

    if all_ok:
        print("\n✅ Tokenizer ready.")
    else:
        print("\n⚠️  Some round-trip checks failed — check corpus quality.")

    # Compression ratio
    total_chars = total_tokens = 0
    for sent in test_sentences:
        total_chars  += len(sent)
        total_tokens += len(sp.encode(sent))
    print(f"Avg chars/token: {total_chars/max(1, total_tokens):.2f}")


if __name__ == "__main__":
    main()
