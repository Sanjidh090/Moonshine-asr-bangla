"""
train.py
────────
Adapts UsefulSensors/moonshine-base for Bengali by swapping its
embedding table and LM head to match the Bengali SentencePiece vocab,
freezes the encoder, and fine-tunes the decoder on the train/dev TSV
splits produced by prepare_data.py.

Best checkpoint (lowest val loss) is saved to SAVE_DIR/best/.

Usage:
    python scripts/train.py

Key env-var overrides:
    LIPI_GHOR   – path that contains train.tsv, dev.tsv and audio/
    SAVE_DIR    – where checkpoints are written
    SP_MODEL    – path to bn_unigram.model
    HF_MODEL    – HuggingFace model id (default UsefulSensors/moonshine-base)
    EPOCHS      – number of training epochs        (default 15)
    BATCH_SIZE  – batch size per GPU/CPU step      (default 8)
    LR          – initial learning rate             (default 2e-4)
"""

import os
import shutil
import time
from pathlib import Path

import numpy as np
import sentencepiece as spm
import soundfile as sf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSpeechSeq2Seq

# ── CONFIG ────────────────────────────────────────────────────────────────────
LIPI_GHOR   = Path(os.getenv("LIPI_GHOR",  "/kaggle/working/lipi-ghor"))
SAVE_DIR    = Path(os.getenv("SAVE_DIR",   "/kaggle/working/checkpoints"))
SP_MODEL    = Path(os.getenv("SP_MODEL",   str(LIPI_GHOR / "tokenizer" / "bn_unigram.model")))
HF_MODEL    = os.getenv("HF_MODEL",        "UsefulSensors/moonshine-base")

BATCH_SIZE  = int(os.getenv("BATCH_SIZE",  "8"))
EPOCHS      = int(os.getenv("EPOCHS",      "15"))
LR          = float(os.getenv("LR",        "2e-4"))
LOG_EVERY   = int(os.getenv("LOG_EVERY",   "50"))

SAMPLE_RATE = 16_000
MAX_AUDIO   = 30.0      # seconds
MAX_TOKENS  = 194
# ─────────────────────────────────────────────────────────────────────────────


# ── Dataset ───────────────────────────────────────────────────────────────────
class BengaliASRDataset(Dataset):
    def __init__(self, tsv_path: Path, audio_base: Path):
        self.audio_base = audio_base
        self.samples: list[tuple[str, str]] = []
        with open(tsv_path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    self.samples.append((parts[0], parts[1]))
        print(f"  {len(self.samples):,} samples — {tsv_path.name}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        rel, transcript = self.samples[idx]
        try:
            audio, _ = sf.read(
                str(self.audio_base / rel), dtype="float32", always_2d=False
            )
        except Exception:
            return None
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if len(audio) / SAMPLE_RATE > MAX_AUDIO:
            return None
        ids = [sp.bos_id()] + sp.encode_as_ids(transcript) + [sp.eos_id()]
        if len(ids) > MAX_TOKENS:
            return None
        return {
            "audio":     torch.tensor(audio, dtype=torch.float32),
            "input_ids": torch.tensor(ids,   dtype=torch.long),
        }


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    max_a = ((max(b["audio"].shape[0] for b in batch) + 159) // 160) * 160
    audio_b = torch.zeros(len(batch), max_a)
    for i, b in enumerate(batch):
        audio_b[i, : b["audio"].shape[0]] = b["audio"]
    max_t = max(b["input_ids"].shape[0] for b in batch)
    token_b = torch.full((len(batch), max_t), -100, dtype=torch.long)
    for i, b in enumerate(batch):
        token_b[i, : b["input_ids"].shape[0]] = b["input_ids"]
    return {"audio": audio_b, "input_ids": token_b}


# ── Training helpers ──────────────────────────────────────────────────────────
def train_epoch(epoch: int) -> float:
    model.train()
    total_loss, t0 = 0.0, time.time()
    for step, batch in enumerate(train_loader):
        if batch is None:
            continue
        audio     = batch["audio"].to(device)
        input_ids = batch["input_ids"].to(device)
        dec_input = input_ids[:, :-1].clone()
        labels    = input_ids[:, 1:].clone()
        dec_input[dec_input == -100] = 0

        optimizer.zero_grad()
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
            out  = model(input_values=audio, decoder_input_ids=dec_input)
            loss = nn.functional.cross_entropy(
                out.logits.reshape(-1, out.logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0
        )
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()

        if (step + 1) % LOG_EVERY == 0:
            avg = total_loss / (step + 1)
            print(
                f"  ep{epoch} step{step+1}/{len(train_loader)} "
                f"loss={avg:.4f} t={time.time()-t0:.0f}s"
            )
    return total_loss / max(1, len(train_loader))


def greedy_wer(hyps: list[str], refs: list[str]) -> float:
    total_w = total_e = 0
    for h, r in zip(hyps, refs):
        h_words, r_words = h.split(), r.split()
        total_w += len(r_words)
        d = list(range(len(r_words) + 1))
        for hw in h_words:
            p, d[0] = d[:], d[0] + 1
            for j, rw in enumerate(r_words):
                d[j + 1] = min(p[j] + (hw != rw), d[j] + 1, p[j + 1] + 1)
        total_e += d[len(r_words)]
    return total_e / max(1, total_w)


@torch.no_grad()
def validate() -> tuple[float, float]:
    model.eval()
    total_loss, n = 0.0, 0
    hyps: list[str] = []
    refs: list[str] = []

    for batch in val_loader:
        if batch is None:
            continue
        audio     = batch["audio"].to(device)
        input_ids = batch["input_ids"].to(device)
        dec_input = input_ids[:, :-1].clone()
        labels    = input_ids[:, 1:].clone()
        dec_input[dec_input == -100] = 0

        with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
            out  = model(input_values=audio, decoder_input_ids=dec_input)
            loss = nn.functional.cross_entropy(
                out.logits.reshape(-1, out.logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )
        total_loss += loss.item()
        n += 1

        if len(hyps) < 32:
            for ids, ref_ids in zip(out.logits.argmax(-1), input_ids):
                ref_ids = ref_ids[ref_ids != -100].tolist()[1:-1]
                hyps.append(sp.decode_ids(ids.tolist()))
                refs.append(sp.decode_ids(ref_ids))

    model.train()
    return total_loss / max(1, n), greedy_wer(hyps, refs)


def save_checkpoint(val_loss: float):
    best_dir = SAVE_DIR / "best"
    best_dir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), best_dir / "model_state.pt")
    model.config.save_pretrained(best_dir)
    shutil.copy(SP_MODEL, best_dir / SP_MODEL.name)
    print(f"  ✅ Checkpoint saved  val_loss={val_loss:.4f}  →  {best_dir}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    global sp, model, device, use_amp
    global train_loader, val_loader, optimizer, scaler

    # Tokenizer
    sp = spm.SentencePieceProcessor()
    sp.load(str(SP_MODEL))
    bn_vocab = sp.get_piece_size()
    print(f"Vocab size: {bn_vocab}")

    # Device
    device  = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"
    SAVE_DIR.mkdir(exist_ok=True)
    print(f"Device: {device}  |  AMP: {use_amp}")

    # Load base model and swap vocabulary
    print(f"Loading base model: {HF_MODEL}")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        HF_MODEL, trust_remote_code=True, torch_dtype=torch.float32
    )
    old_embed = model.model.decoder.embed_tokens
    embed_dim = old_embed.embedding_dim

    new_embed = nn.Embedding(bn_vocab, embed_dim)
    nn.init.normal_(new_embed.weight, mean=0.0, std=embed_dim ** -0.5)

    new_proj = nn.Linear(embed_dim, bn_vocab, bias=False)
    new_proj.weight = new_embed.weight      # tied weights

    model.model.decoder.embed_tokens = new_embed
    model.proj_out                   = new_proj
    model.config.vocab_size          = bn_vocab
    print(f"Vocab swapped: {old_embed.num_embeddings} → {bn_vocab}")

    # Freeze encoder
    for p in model.model.encoder.parameters():
        p.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    total     = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Trainable: {trainable:.1f}M / {total:.1f}M")
    model = model.to(device)

    # Datasets & loaders
    train_ds = BengaliASRDataset(LIPI_GHOR / "train.tsv", LIPI_GHOR)
    val_ds   = BengaliASRDataset(LIPI_GHOR / "dev.tsv",   LIPI_GHOR)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=2,
    )
    print(f"Train batches: {len(train_loader)}  |  Val batches: {len(val_loader)}")

    # Optimizer, scheduler, scaler
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=1e-2,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-5
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Training loop
    best_val = float("inf")
    for epoch in range(1, EPOCHS + 1):
        train_loss        = train_epoch(epoch)
        val_loss, wer_val = validate()
        scheduler.step()
        print(
            f"\nepoch {epoch}  train={train_loss:.4f}  val={val_loss:.4f}  "
            f"WER={wer_val*100:.1f}%  lr={scheduler.get_last_lr()[0]:.2e}"
        )
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(val_loss)
        print()

    print(f"Training complete. Best val loss: {best_val:.4f}")


if __name__ == "__main__":
    main()
