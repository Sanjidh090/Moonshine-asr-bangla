"""
transcribe.py
─────────────
Loads a fine-tuned Bengali Moonshine checkpoint and transcribes audio.

Supports two modes:
  • single  – transcribe one short clip (≤30 s) at full quality
  • longform – slide a 30-s window over a long recording with 25-s stride

Usage examples:
    # Single clip
    python scripts/transcribe.py --audio clip.wav

    # Long-form audio
    python scripts/transcribe.py --audio long.wav --mode longform

    # Use a custom checkpoint directory
    python scripts/transcribe.py \
        --audio clip.wav \
        --checkpoint /kaggle/working/checkpoints/best \
        --sp-model /kaggle/working/lipi-ghor/tokenizer/bn_unigram.model
"""

import argparse
import unicodedata
from pathlib import Path

import numpy as np
import sentencepiece as spm
import soundfile as sf
import torch
import torch.nn as nn
from transformers import AutoModelForSpeechSeq2Seq

# ── defaults (can be overridden via CLI) ──────────────────────────────────────
DEFAULT_CHECKPOINT = "/kaggle/working/checkpoints/best"
DEFAULT_SP_MODEL   = "/kaggle/working/lipi-ghor/tokenizer/bn_unigram.model"
SAMPLE_RATE        = 16_000
MAX_NEW_TOKENS     = 448
CHUNK_SEC          = 30
STEP_SEC           = 25
REPETITION_LIMIT   = 3   # suppress tokens seen ≥ this many times
# ─────────────────────────────────────────────────────────────────────────────


def load_audio(path: str | Path) -> np.ndarray:
    """Load a WAV/FLAC file, convert to mono 16 kHz float32."""
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        try:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
        except ImportError:
            raise RuntimeError(
                f"Audio sample rate is {sr} Hz but librosa is not installed. "
                "Install it with: pip install librosa"
            )
    return audio


def load_model(checkpoint: str | Path, sp: spm.SentencePieceProcessor, device: str):
    """Load Moonshine from a fine-tuned checkpoint directory."""
    checkpoint = Path(checkpoint)
    bn_vocab   = sp.get_piece_size()

    print(f"Loading model from: {checkpoint}")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        str(checkpoint), trust_remote_code=True, torch_dtype=torch.float32
    )

    # Re-attach Bengali vocabulary if the checkpoint vocab differs
    if model.proj_out.out_features != bn_vocab:
        print(f"  Resizing vocab: {model.proj_out.out_features} → {bn_vocab}")
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


def _greedy_decode(
    audio_np: np.ndarray,
    model,
    sp: spm.SentencePieceProcessor,
    device: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> str:
    """Greedy autoregressive decode for a single audio chunk."""
    audio_t = torch.tensor(audio_np, dtype=torch.float32).unsqueeze(0).to(device)
    generated = [sp.bos_id()]

    with torch.no_grad():
        enc_out    = model.model.encoder(input_values=audio_t)
        enc_hidden = enc_out.last_hidden_state

        for _ in range(max_new_tokens):
            dec_in  = torch.tensor([generated], dtype=torch.long, device=device)
            dec_out = model.model.decoder(
                input_ids=dec_in,
                encoder_hidden_states=enc_hidden,
            )
            logits = model.proj_out(dec_out.last_hidden_state[:, -1, :])

            # Repetition penalty
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


def transcribe_single(audio_np: np.ndarray, model, sp, device: str) -> str:
    dur = len(audio_np) / SAMPLE_RATE
    print(f"Audio duration: {dur:.2f}s")
    text = _greedy_decode(audio_np, model, sp, device)
    return unicodedata.normalize("NFC", text)


def transcribe_longform(
    audio_np: np.ndarray,
    model,
    sp: spm.SentencePieceProcessor,
    device: str,
    chunk_sec: int = CHUNK_SEC,
    step_sec:  int = STEP_SEC,
) -> str:
    chunk_size = chunk_sec * SAMPLE_RATE
    step_size  = step_sec  * SAMPLE_RATE
    total_min  = len(audio_np) / SAMPLE_RATE / 60

    parts: list[str] = []
    for start in range(0, len(audio_np), step_size):
        end   = start + chunk_size
        chunk = audio_np[start:end]
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)))

        text = _greedy_decode(chunk, model, sp, device, max_new_tokens=256)
        parts.append(text)
        print(
            f"  Progress: {start/SAMPLE_RATE/60:.2f} / {total_min:.2f} min",
            end="\r",
        )

    print()
    full_text = " ".join(parts)
    return unicodedata.normalize("NFC", full_text)


def main():
    parser = argparse.ArgumentParser(description="Bengali Moonshine transcription")
    parser.add_argument("--audio",      required=True, help="Path to audio file")
    parser.add_argument(
        "--mode", choices=["single", "longform"], default="single",
        help="Transcription mode (default: single)",
    )
    parser.add_argument(
        "--checkpoint", default=DEFAULT_CHECKPOINT,
        help="Path to fine-tuned checkpoint directory",
    )
    parser.add_argument(
        "--sp-model", default=DEFAULT_SP_MODEL,
        help="Path to SentencePiece .model file",
    )
    parser.add_argument(
        "--chunk-sec", type=int, default=CHUNK_SEC,
        help="Chunk length in seconds for longform mode (default: 30)",
    )
    parser.add_argument(
        "--step-sec", type=int, default=STEP_SEC,
        help="Step size in seconds for longform mode (default: 25)",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load tokenizer
    sp = spm.SentencePieceProcessor()
    sp.load(args.sp_model)
    print(f"Tokenizer loaded: vocab={sp.get_piece_size()}")

    # Load model
    model = load_model(args.checkpoint, sp, device)

    # Load audio
    print(f"Loading audio: {args.audio}")
    audio = load_audio(args.audio)

    # Transcribe
    if args.mode == "longform":
        result = transcribe_longform(
            audio, model, sp, device,
            chunk_sec=args.chunk_sec,
            step_sec=args.step_sec,
        )
    else:
        result = transcribe_single(audio, model, sp, device)

    print("\n── Transcription ──────────────────────────────────────────────")
    print(result)
    print("───────────────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
