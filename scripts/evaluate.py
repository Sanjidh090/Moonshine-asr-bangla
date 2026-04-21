"""
evaluate.py
───────────
Computes Word Error Rate (WER) and Character Error Rate (CER) for a
fine-tuned Bengali Moonshine model against a reference annotation file
or a TSV split (file_path <TAB> transcript).

Usage examples:
    # Evaluate a single audio file against a reference text file
    python scripts/evaluate.py \
        --audio  path/to/audio.wav \
        --ref    path/to/reference.txt

    # Evaluate the full test split
    python scripts/evaluate.py \
        --tsv    /kaggle/working/lipi-ghor/test.tsv \
        --audio-base /kaggle/working/lipi-ghor

    # Use a custom checkpoint
    python scripts/evaluate.py \
        --tsv  test.tsv \
        --audio-base /data/audio \
        --checkpoint /kaggle/working/checkpoints/best \
        --sp-model   /kaggle/working/lipi-ghor/tokenizer/bn_unigram.model
"""

import argparse
import unicodedata
from pathlib import Path

import numpy as np
import sentencepiece as spm
import soundfile as sf
import torch
import torch.nn as nn
from jiwer import cer, wer
from transformers import AutoModelForSpeechSeq2Seq

# ── defaults ──────────────────────────────────────────────────────────────────
DEFAULT_CHECKPOINT = "/kaggle/working/checkpoints/best"
DEFAULT_SP_MODEL   = "/kaggle/working/lipi-ghor/tokenizer/bn_unigram.model"
SAMPLE_RATE        = 16_000
MAX_NEW_TOKENS     = 448
CHUNK_SEC          = 30
STEP_SEC           = 25
REPETITION_LIMIT   = 3
# ─────────────────────────────────────────────────────────────────────────────


def load_audio(path: str | Path) -> np.ndarray:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    return audio


def load_model(checkpoint: str | Path, sp: spm.SentencePieceProcessor, device: str):
    checkpoint = Path(checkpoint)
    bn_vocab   = sp.get_piece_size()

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        str(checkpoint), trust_remote_code=True, torch_dtype=torch.float32
    )

    if model.proj_out.out_features != bn_vocab:
        embed_dim = model.model.decoder.embed_tokens.embedding_dim
        new_embed = nn.Embedding(bn_vocab, embed_dim)
        nn.init.normal_(new_embed.weight, mean=0.0, std=embed_dim ** -0.5)
        new_proj = nn.Linear(embed_dim, bn_vocab, bias=False)
        new_proj.weight = new_embed.weight
        model.model.decoder.embed_tokens = new_embed
        model.proj_out                   = new_proj

        state = torch.load(checkpoint / "model_state.pt", map_location="cpu")
        model.load_state_dict(state, strict=False)

    model = model.to(device)
    model.eval()
    return model


def _greedy_decode(audio_np: np.ndarray, model, sp, device: str) -> str:
    audio_t = torch.tensor(audio_np, dtype=torch.float32).unsqueeze(0).to(device)
    generated = [sp.bos_id()]

    with torch.no_grad():
        enc_hidden = model.model.encoder(input_values=audio_t).last_hidden_state
        for _ in range(MAX_NEW_TOKENS):
            dec_in  = torch.tensor([generated], dtype=torch.long, device=device)
            dec_out = model.model.decoder(
                input_ids=dec_in,
                encoder_hidden_states=enc_hidden,
            )
            logits = model.proj_out(dec_out.last_hidden_state[:, -1, :])
            tok_ids, counts = torch.unique(
                torch.tensor(generated), return_counts=True
            )
            for tok_id, count in zip(tok_ids.tolist(), counts.tolist()):
                if count >= REPETITION_LIMIT:
                    logits[0, tok_id] = -1e9
            next_tok = logits.argmax(-1).item()
            if next_tok == sp.eos_id():
                break
            generated.append(next_tok)

    pieces = [sp.id_to_piece(t) for t in generated[1:]]
    return "".join(pieces).replace("▁", " ").strip()


def _transcribe_longform(audio_np: np.ndarray, model, sp, device: str) -> str:
    chunk_size = CHUNK_SEC * SAMPLE_RATE
    step_size  = STEP_SEC  * SAMPLE_RATE
    parts: list[str] = []
    for start in range(0, len(audio_np), step_size):
        chunk = audio_np[start : start + chunk_size]
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
        parts.append(_greedy_decode(chunk, model, sp, device))
    return " ".join(parts)


def normalise(text: str) -> str:
    return unicodedata.normalize("NFC", text.strip())


def print_metrics(label: str, hypothesis: str, reference: str):
    word_err = wer(reference, hypothesis)
    char_err = cer(reference, hypothesis)
    print(f"\n── Evaluation: {label} ──────────────────────────────────────")
    print(f"  WER : {word_err*100:.2f}%")
    print(f"  CER : {char_err*100:.2f}%")
    print("─" * 60)
    return word_err, char_err


def main():
    parser = argparse.ArgumentParser(description="Bengali Moonshine evaluation")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--audio", help="Single audio file to evaluate")
    group.add_argument("--tsv",   help="TSV split file (rel_path<TAB>transcript)")

    parser.add_argument("--ref",        help="Reference text file (used with --audio)")
    parser.add_argument("--audio-base", default=".",
                        help="Base directory for relative paths in the TSV")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--sp-model",   default=DEFAULT_SP_MODEL)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit evaluation to N samples (useful for quick checks)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    sp = spm.SentencePieceProcessor()
    sp.load(args.sp_model)

    model = load_model(args.checkpoint, sp, device)
    print("Model loaded.")

    # ── Single file mode ──────────────────────────────────────────────────────
    if args.audio:
        if not args.ref:
            parser.error("--ref is required when using --audio")
        audio = load_audio(args.audio)
        dur   = len(audio) / SAMPLE_RATE
        auto_mode = "longform" if dur > CHUNK_SEC else "single"
        print(f"Audio: {dur:.1f}s  →  mode: {auto_mode}")

        if auto_mode == "longform":
            hyp = _transcribe_longform(audio, model, sp, device)
        else:
            hyp = _greedy_decode(audio, model, sp, device)

        hyp = normalise(hyp)
        ref = normalise(Path(args.ref).read_text(encoding="utf-8"))

        print("\nREF:", ref[:200])
        print("HYP:", hyp[:200])
        print_metrics(Path(args.audio).name, hyp, ref)
        return

    # ── TSV batch mode ────────────────────────────────────────────────────────
    tsv_path   = Path(args.tsv)
    audio_base = Path(args.audio_base)

    samples: list[tuple[str, str]] = []
    with open(tsv_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                samples.append((parts[0], parts[1]))

    if args.max_samples:
        samples = samples[: args.max_samples]

    print(f"Evaluating {len(samples)} samples from {tsv_path.name} ...")

    all_hyps: list[str] = []
    all_refs: list[str] = []
    errors: int = 0

    for i, (rel, ref_text) in enumerate(samples, 1):
        audio_path = audio_base / rel
        try:
            audio = load_audio(audio_path)
        except Exception as e:
            print(f"  [{i}/{len(samples)}] SKIP {rel}: {e}")
            errors += 1
            continue

        dur  = len(audio) / SAMPLE_RATE
        auto = "longform" if dur > CHUNK_SEC else "single"
        if auto == "longform":
            hyp = _transcribe_longform(audio, model, sp, device)
        else:
            hyp = _greedy_decode(audio, model, sp, device)

        all_hyps.append(normalise(hyp))
        all_refs.append(normalise(ref_text))

        if i % 10 == 0 or i == len(samples):
            running_wer = wer(all_refs, all_hyps)
            running_cer = cer(all_refs, all_hyps)
            print(
                f"  [{i}/{len(samples)}]  running WER={running_wer*100:.2f}%"
                f"  CER={running_cer*100:.2f}%"
            )

    if all_hyps:
        final_wer, final_cer = print_metrics(
            tsv_path.name, all_hyps, all_refs
        )

    if errors:
        print(f"\n  ⚠️  Skipped {errors} samples due to load errors.")


if __name__ == "__main__":
    main()
