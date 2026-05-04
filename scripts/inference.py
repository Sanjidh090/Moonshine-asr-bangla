import torch
import librosa
import numpy as np
from transformers import AutoTokenizer, AutoModelForSpeechSeq2Seq

# --- Configuration ---
MODEL_ID    = "Sanji27/fountain_base_15ep"
TOK_ID      = MODEL_ID # Ensure this matches your Phase 2 tokenizer!

SAMPLE_RATE = 16_000
CHUNK_SEC   = 15.0  # 15 second chunks (gives the model plenty of context)
MAX_TOKENS  = 190
NUM_BEAMS   = 4

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype  = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32

print(f"Loading model to {device}... hang tight!")

# Load Model & Tokenizer
tokenizer = AutoTokenizer.from_pretrained(TOK_ID, trust_remote_code=True)
model     = AutoModelForSpeechSeq2Seq.from_pretrained(
    MODEL_ID, trust_remote_code=True, torch_dtype=dtype
).to(device).eval()

BOS_ID = model.config.decoder_start_token_id or 1
EOS_ID = model.config.eos_token_id or 2

@torch.no_grad()
def transcribe(file_path):
    print(f"Loading audio into memory: {file_path}")
    
    try:
        audio, _ = librosa.load(file_path, sr=SAMPLE_RATE)
    except Exception as e:
        print(f"Error loading audio: {e}")
        return

    # ============================================================
    # THE FIX: Slice into clean 15-second blocks, no strict VAD
    # ============================================================
    chunk_samples = int(SAMPLE_RATE * CHUNK_SEC)
    segments = [audio[i:i + chunk_samples] for i in range(0, len(audio), chunk_samples)]

    if not segments:
        print("No audio found.")
        return

    full_transcription = []
    print(f"\n--- Transcribing {len(segments)} chunks LIVE (15s each) ---\n")
    
    for i, seg in enumerate(segments):
        # Skip weird microscopic leftovers at the very end of the file
        if len(seg) < 1600:
            continue

        # Pad to 160 alignment for Moonshine
        r = len(seg) % 160
        if r:
            seg = np.concatenate([seg, np.zeros(160 - r, dtype=np.float32)])

        audio_t = torch.tensor(seg, dtype=dtype).unsqueeze(0).to(device)
        
        output_ids = model.generate(
            input_values=audio_t,
            decoder_input_ids=torch.tensor([[BOS_ID]], device=device),
            max_new_tokens=MAX_TOKENS,
            num_beams=NUM_BEAMS,
            repetition_penalty=2.5,     # High penalty to stop infinite numbers
            no_repeat_ngram_size=3,     # Stop repeating phrases
            early_stopping=True,        # Stop exactly when sentence ends
            eos_token_id=EOS_ID,
            pad_token_id=0,
        )
        
        tokens = output_ids[0].tolist()
        if tokens and tokens[0] == BOS_ID:
            tokens = tokens[1:]
            
        text = tokenizer.decode(tokens, skip_special_tokens=True).strip()
        
        if text:
            print(text, end=" ", flush=True)
            full_transcription.append(text)

    print("\n\n--- Done ---")
    return " ".join(full_transcription)

if __name__ == "__main__":
    # Test with your MP3
    test_file = '/kaggle/input/datasets/sanjidh090/dl-sprint-4-0-bengali-long-form-asr/transcription/train/audio/train_002.wav'
    transcribe(test_file)