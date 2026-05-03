"""
train_base.py
=============
Bengali Moonshine BASE fine-tune
  - UsefulSensors/moonshine-base  (61.5M params)
  - Same pipeline as tiny, adjusted for larger model
  - BATCH_SIZE=4, GRAD_ACCUM=8 → effective batch=32 (same as paper)
  - LR=2e-5 (same as Flavors paper)
"""

import os, csv, time, random
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSpeechSeq2Seq
import schedulefree

# ═══════════════════════════════════════════════════════════════════════════════
#  PATHS
# ═══════════════════════════════════════════════════════════════════════════════
TRAINING_ROOT = Path(r"D:\Kminds_Sanjidh090_dataset\Lipighor_wavs")
WAVS_DIR      = TRAINING_ROOT / "wavs_asr_chunks" / "wavs"
META_CSV      = TRAINING_ROOT / "wavs_asr_chunks" / "metadata.csv"

WORK_DIR      = Path(r"F:\Sanjid_2203090_Kminds\moonshine-bn-base")
SAVE_DIR      = WORK_DIR / "checkpoints"

for d in [WORK_DIR, SAVE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
HF_MODEL    = "UsefulSensors/moonshine-base"
HF_TOKEN    = "hf_"

SAMPLE_RATE    = 16_000
MAX_AUDIO_SEC  = 30.0
MIN_AUDIO_SEC  = 4.0
MAX_TOKENS     = 448

BATCH_SIZE     = 4
GRAD_ACCUM     = 8
EPOCHS         = 21
LR             = 2e-5
LOG_EVERY      = 50
PATIENCE       = 4

NUM_WORKERS    = 0

FILLER_WORDS   = {"মিউজিক", "প্রশংসা"}

device   = "cuda" if torch.cuda.is_available() else "cpu"
use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
use_fp16 = device == "cuda" and not use_bf16
dtype    = torch.bfloat16 if use_bf16 else torch.float16 if use_fp16 else torch.float32

print("=" * 60)
print(f"  Model   : {HF_MODEL}  (61.5M params)")
print(f"  Device  : {device}  |  dtype: {dtype}")
if device == "cuda":
    props = torch.cuda.get_device_properties(0)
    print(f"  GPU     : {props.name}")
    print(f"  VRAM    : {props.total_memory / 1e9:.1f} GB")
print(f"  Eff.batch: {BATCH_SIZE * GRAD_ACCUM}  (paper: 32)")
print(f"  LR      : {LR}")
print("=" * 60)


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — Read metadata + write TSVs
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[1/4] Reading metadata.csv ...")

rows = []
with open(META_CSV, encoding="utf-8") as f:
    for row in csv.DictReader(f):
        try:
            dur   = float(row["duration"])
            text  = row["text"].strip()
            fname = Path(row["file_name"]).name
            wav   = WAVS_DIR / fname

            words        = text.split()
            filler_count = sum(1 for w in words if w in FILLER_WORDS)
            if filler_count / max(1, len(words)) > 0.5:
                continue

            if (MIN_AUDIO_SEC <= dur <= MAX_AUDIO_SEC
                    and len(text) >= 3
                    and wav.exists()):
                rows.append({"wav": wav, "text": text, "dur": dur})
        except (ValueError, KeyError):
            pass

print(f"   ✓ {len(rows):,} valid rows (filler filtered)")

if not (WORK_DIR / "train.tsv").exists():
    random.seed(42)
    random.shuffle(rows)
    n = len(rows)
    splits = {
        "train": rows[:int(n * 0.90)],
        "dev":   rows[int(n * 0.90): int(n * 0.95)],
        "test":  rows[int(n * 0.95):],
    }
    for name, split_rows in splits.items():
        with open(WORK_DIR / f"{name}.tsv", "w", encoding="utf-8") as f:
            for r in split_rows:
                f.write(f"{r['wav']}\t{r['text']}\n")
        hrs = sum(r["dur"] for r in split_rows) / 3600
        print(f"   {name:<8} {len(split_rows):>6,} utterances   {hrs:.1f}h")
else:
    print("   ✓ TSV files already exist — skipping split")


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — Load tokenizer + model
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n[2/4] Loading {HF_MODEL} ...")
os.environ["HF_TOKEN"] = HF_TOKEN

tokenizer = AutoTokenizer.from_pretrained(
    HF_MODEL, token=HF_TOKEN, trust_remote_code=True
)
print(f"   ✓ Tokenizer — vocab: {tokenizer.vocab_size}")

for test_str in ["আমার সোনার বাংলা", "মানসিক রোগ সম্পর্কে সচেতন হওয়া জরুরি"]:
    ids  = tokenizer.encode(test_str, add_special_tokens=False)
    back = tokenizer.decode(ids)
    assert back == test_str, f"Round-trip failed: {test_str!r} → {back!r}"
    print(f"   ✓ round-trip OK ({len(ids)} tokens): '{test_str[:30]}'")

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    HF_MODEL, trust_remote_code=True, dtype=torch.float32, token=HF_TOKEN
)

BOS_ID = model.config.decoder_start_token_id or 1
EOS_ID = model.config.eos_token_id or 2
# FIX: hardcode PAD_ID=0 — model.config.pad_token_id returns 2 (same as EOS),
#      which would corrupt teacher-forcing wherever -100 padding exists.
PAD_ID = 0
print(f"   ✓ Token IDs — BOS:{BOS_ID}  EOS:{EOS_ID}  PAD:{PAD_ID}")

model.gradient_checkpointing_enable()

total     = sum(p.numel() for p in model.parameters()) / 1e6
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
print(f"   ✓ {total:.1f}M params  {trainable:.1f}M trainable")
print(f"   ✓ Gradient checkpointing enabled")

model = model.to(device)


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — Dataset + DataLoader
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[3/4] Building dataloaders ...")


class BengaliASRDataset(Dataset):
    def __init__(self, tsv_path):
        self.samples = []
        with open(tsv_path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t", 1)
                if len(parts) == 2:
                    self.samples.append((Path(parts[0]), parts[1]))
        print(f"   {len(self.samples):,} samples — {Path(tsv_path).name}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        try:
            wav_path, transcript = self.samples[idx]
            audio, _ = sf.read(str(wav_path), dtype="float32", always_2d=False)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)

            dur = len(audio) / SAMPLE_RATE
            if not (MIN_AUDIO_SEC <= dur <= MAX_AUDIO_SEC):
                return None

            remainder = len(audio) % 160
            if remainder:
                audio = np.concatenate(
                    [audio, np.zeros(160 - remainder, dtype=np.float32)]
                )

            ids = tokenizer.encode(transcript, add_special_tokens=False)
            # FIX: removed `min(MAX_TOKENS - 2, max_tok)` — the old TOKENS_PER_SEC=6
            #      heuristic assumed English (~1.5 tok/word). Bengali tokenizes at
            #      ~7 tok/word so it filtered every single sample. Just cap at MAX_TOKENS.
            if len(ids) == 0 or len(ids) > MAX_TOKENS - 2:
                return None

            ids = [BOS_ID] + ids + [EOS_ID]
            return {
                "audio":     torch.tensor(audio, dtype=torch.float32),
                "input_ids": torch.tensor(ids,   dtype=torch.long),
            }
        except Exception:
            return None


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    max_a   = ((max(b["audio"].shape[0] for b in batch) + 159) // 160) * 160
    audio_b = torch.zeros(len(batch), max_a)
    for i, b in enumerate(batch):
        audio_b[i, :b["audio"].shape[0]] = b["audio"]
    max_t   = max(b["input_ids"].shape[0] for b in batch)
    token_b = torch.full((len(batch), max_t), -100, dtype=torch.long)
    for i, b in enumerate(batch):
        token_b[i, :b["input_ids"].shape[0]] = b["input_ids"]
    return {"audio": audio_b, "input_ids": token_b}


train_ds = BengaliASRDataset(WORK_DIR / "train.tsv")
val_ds   = BengaliASRDataset(WORK_DIR / "dev.tsv")
test_ds  = BengaliASRDataset(WORK_DIR / "test.tsv")

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True,
    collate_fn=collate_fn, num_workers=NUM_WORKERS,
    pin_memory=(device == "cuda"),
)
val_loader = DataLoader(
    val_ds, batch_size=BATCH_SIZE, shuffle=False,
    collate_fn=collate_fn, num_workers=NUM_WORKERS,
)

# Sanity-check: probe first 200 samples so failures are self-explanatory
batch = None
for b in train_loader:
    if b is not None:
        batch = b
        break

if batch is None:
    passed = failed_dur = failed_tok = failed_read = 0
    for wav_path, text in train_ds.samples[:200]:
        try:
            audio, _ = sf.read(str(wav_path), dtype="float32", always_2d=False)
            dur = len(audio) / SAMPLE_RATE
            if not (MIN_AUDIO_SEC <= dur <= MAX_AUDIO_SEC):
                failed_dur += 1
                continue
            ids = tokenizer.encode(text, add_special_tokens=False)
            if len(ids) == 0 or len(ids) > MAX_TOKENS - 2:
                failed_tok += 1
                continue
            passed += 1
        except Exception:
            failed_read += 1
    print(f"   ⚠ Probe (200 samples): passed={passed}  "
          f"dur_fail={failed_dur}  tok_fail={failed_tok}  read_fail={failed_read}")

assert batch is not None, "All batches returned None — check probe counts above"
print(f"   ✓ Audio  : {batch['audio'].shape}")
print(f"   ✓ Tokens : {batch['input_ids'].shape}")
print(f"   ✓ Train  : {len(train_loader)} batches  "
      f"({len(train_loader) // GRAD_ACCUM} steps/epoch)")
print(f"   ✓ Val    : {len(val_loader)} batches")


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 4 — Train
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[4/4] Training ...")

optimizer = schedulefree.AdamWScheduleFree(
    model.parameters(),
    lr=LR,
    betas=(0.9, 0.999),
    weight_decay=1e-2,
    warmup_steps=500,
)

scaler    = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
amp_dtype = dtype if device == "cuda" else torch.float32

RESUME_PATH = SAVE_DIR / "best" / "training_state.pt"
start_epoch = 1
BEST_VAL    = float("inf")
BEST_WER    = float("inf")
patience_ct = 0

if RESUME_PATH.exists():
    print("   Resuming from checkpoint ...")
    state = torch.load(RESUME_PATH, map_location=device)
    model.load_state_dict(
        torch.load(SAVE_DIR / "best" / "model_state.pt", map_location=device)
    )
    optimizer.load_state_dict(state["optimizer"])
    start_epoch = state["epoch"] + 1
    BEST_VAL    = state["val_loss"]
    BEST_WER    = state["wer"]
    patience_ct = state.get("patience", 0)
    print(f"   ✓ Resumed from epoch {state['epoch']}  "
          f"val_loss={BEST_VAL:.4f}  WER={BEST_WER*100:.1f}%")


def greedy_wer(hyps, refs):
    total_w = total_e = 0
    for h, r in zip(hyps, refs):
        h, r = h.split(), r.split()
        total_w += len(r)
        d = list(range(len(r) + 1))
        for hc in h:
            p, d[0] = d[:], d[0] + 1
            for j, rc in enumerate(r):
                d[j+1] = min(p[j] + (hc != rc), d[j] + 1, p[j+1] + 1)
        total_e += d[len(r)]
    return total_e / max(1, total_w)


def train_epoch(epoch):
    optimizer.train()
    model.train()
    total_loss, t0 = 0.0, time.time()
    optimizer.zero_grad()
    steps = 0

    for step, batch in enumerate(train_loader):
        if batch is None:
            continue
        audio     = batch["audio"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        dec_input = input_ids[:, :-1].clone()
        labels    = input_ids[:, 1:].clone()
        dec_input[dec_input == -100] = PAD_ID

        with torch.autocast(device_type=device, dtype=amp_dtype,
                            enabled=(device == "cuda")):
            out  = model(input_values=audio, decoder_input_ids=dec_input)
            loss = nn.functional.cross_entropy(
                out.logits.reshape(-1, out.logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            ) / GRAD_ACCUM

        scaler.scale(loss).backward()
        total_loss += loss.item() * GRAD_ACCUM

        if (step + 1) % GRAD_ACCUM == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            steps += 1

            if steps % LOG_EVERY == 0:
                avg = total_loss / (step + 1)
                eta = (time.time() - t0) / steps * (
                    len(train_loader) // GRAD_ACCUM - steps
                )
                print(f"   ep{epoch} [{steps:>4}/{len(train_loader)//GRAD_ACCUM}] "
                      f"loss={avg:.4f}  ETA={eta/60:.0f}min")

    return total_loss / max(1, len(train_loader))


@torch.no_grad()
def validate():
    optimizer.eval()
    model.eval()
    total_loss, n = 0.0, 0
    hyps, refs = [], []

    for batch in val_loader:
        if batch is None:
            continue
        audio     = batch["audio"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        dec_input = input_ids[:, :-1].clone()
        labels    = input_ids[:, 1:].clone()
        dec_input[dec_input == -100] = PAD_ID

        with torch.autocast(device_type=device, dtype=amp_dtype,
                            enabled=(device == "cuda")):
            out  = model(input_values=audio, decoder_input_ids=dec_input)
            loss = nn.functional.cross_entropy(
                out.logits.reshape(-1, out.logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )
        total_loss += loss.item()
        n += 1

        if len(hyps) < 64:
            for ids, ref_ids in zip(out.logits.argmax(-1), input_ids):
                ref_ids = ref_ids[ref_ids != -100].tolist()[1:-1]
                hyps.append(tokenizer.decode(ids.tolist(),
                             skip_special_tokens=True))
                refs.append(tokenizer.decode(ref_ids,
                             skip_special_tokens=True))

    return total_loss / max(1, n), greedy_wer(hyps, refs)


def save_checkpoint(epoch, val_loss, wer):
    best_dir = SAVE_DIR / "best"
    best_dir.mkdir(exist_ok=True)
    optimizer.eval()
    model.save_pretrained(best_dir)
    torch.save(model.state_dict(), best_dir / "model_state.pt")
    torch.save({
        "epoch":    epoch,
        "val_loss": val_loss,
        "wer":      wer,
        "patience": patience_ct,
        "optimizer": optimizer.state_dict(),
    }, best_dir / "training_state.pt")
    optimizer.train()
    print(f"   ✅ Saved → epoch={epoch}  "
          f"val_loss={val_loss:.4f}  WER={wer*100:.1f}%")


print(f"\n{'='*60}")
print(f"  Model   : moonshine-base  (61.5M)")
print(f"  Epochs  : {EPOCHS}  |  Patience: {PATIENCE}")
print(f"  Batch   : {BATCH_SIZE} × {GRAD_ACCUM} = {BATCH_SIZE*GRAD_ACCUM} effective")
print(f"  Train   : {len(train_ds):,}  |  Val: {len(val_ds):,}")
print(f"{'='*60}\n")

for epoch in range(start_epoch, EPOCHS + 1):
    t0         = time.time()
    train_loss = train_epoch(epoch)
    val_loss, wer = validate()
    epoch_min  = (time.time() - t0) / 60

    print(f"\nepoch {epoch}/{EPOCHS}  "
          f"train={train_loss:.4f}  val={val_loss:.4f}  "
          f"WER={wer*100:.1f}%  time={epoch_min:.0f}min")

    if val_loss < BEST_VAL:
        BEST_VAL    = val_loss
        BEST_WER    = wer
        patience_ct = 0
        save_checkpoint(epoch, val_loss, wer)
    else:
        patience_ct += 1
        print(f"   ⚠️  No improvement — patience {patience_ct}/{PATIENCE}")
        if patience_ct >= PATIENCE:
            print(f"\n🛑 Early stopping at epoch {epoch}")
            break
    print()

print(f"\n{'='*60}")
print(f"  Training complete!")
print(f"  Best val_loss : {BEST_VAL:.4f}")
print(f"  Best WER      : {BEST_WER*100:.1f}%")
print(f"  Checkpoint    : {SAVE_DIR / 'best'}")
print(f"{'='*60}")
