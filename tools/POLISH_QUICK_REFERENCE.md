# Polish TTS Quick Reference Card

Quick commands for your specific setup with data in `/workspace/10_hours/`.

## Your Directory Structure

```
/workspace/10_hours/
├── transcripts/       # WhisperX JSON files
│   ├── file_001.json
│   ├── file_002.json
│   └── ...
└── source_audio/      # WAV audio files
    ├── file_001.wav
    ├── file_002.wav
    └── ...
```

## Step 1: Train Tokenizer

```bash
# Train Polish BPE tokenizer (with byte-fallback to avoid <unk>)
python tools/tokenizer/train_bpe.py \
    --manifest-dir /workspace/10_hours/transcripts \
    --manifest-format whisperx \
    --language pl \
    --output-prefix checkpoints/polish_bpe \
    --vocab-size 12000 \
    --character-coverage 0.9995 \
    --byte-fallback
```

**Output:** `checkpoints/polish_bpe.model`, `checkpoints/polish_bpe.vocab`

## Step 2: Verify Tokenizer

```bash
# Quick test on sample text
python tools/tokenizer/verify_tokenizer.py \
    --tokenizer checkpoints/polish_bpe.model \
    --language pl \
    --text "wątpię nie było żadnych znaków"

# Comprehensive test on dataset
python tools/tokenizer/verify_tokenizer.py \
    --tokenizer checkpoints/polish_bpe.model \
    --test-dir /workspace/10_hours/transcripts \
    --language pl \
    --sample-size 1000 \
    --show-examples
```

**Expected:** `✓ SUCCESS: No <unk> tokens found!`

## Step 3: Test Audio Discovery

```bash
# Verify audio files can be found (before full preprocessing)
python tools/test_audio_discovery.py \
    --transcript-dir /workspace/10_hours/transcripts \
    --audio-dir /workspace/10_hours/source_audio \
    --sample-size 10
```

**Expected:** `✓ All audio files found!`

## Step 4: Preprocess Dataset

### Option A: Using Helper Script (Easiest)

```bash
bash tools/preprocess_polish.sh \
    /workspace/10_hours/transcripts \
    /workspace/10_hours/source_audio \
    processed_data_polish \
    checkpoints/polish_bpe.model
```

### Option B: Direct Python (More Control)

```bash
python tools/preprocess_polish.py \
    --data-dir /workspace/10_hours/transcripts \
    --audio-dir /workspace/10_hours/source_audio \
    --output-dir processed_data_polish \
    --tokenizer checkpoints/polish_bpe.model \
    --config checkpoints/config.yaml \
    --gpt-checkpoint checkpoints/gpt.pth \
    --device cuda \
    --val-ratio 0.01 \
    --min-duration 0.5 \
    --max-duration 30.0 \
    --skip-existing
```

## Step 5: Check Output

```bash
# View statistics
cat processed_data_polish/stats.json

# Count samples
wc -l processed_data_polish/train_manifest.jsonl
wc -l processed_data_polish/val_manifest.jsonl

# View sample entries
head -n 3 processed_data_polish/train_manifest.jsonl
```

## Complete One-Liner Pipeline

```bash
# Train tokenizer → Verify → Test audio → Preprocess
python tools/tokenizer/train_bpe.py \
    --manifest-dir /workspace/10_hours/transcripts \
    --manifest-format whisperx \
    --language pl \
    --output-prefix checkpoints/polish_bpe \
    --vocab-size 12000 \
    --character-coverage 0.9995 \
    --byte-fallback \
&& python tools/tokenizer/verify_tokenizer.py \
    --tokenizer checkpoints/polish_bpe.model \
    --test-dir /workspace/10_hours/transcripts \
    --language pl \
    --sample-size 1000 \
&& python tools/test_audio_discovery.py \
    --transcript-dir /workspace/10_hours/transcripts \
    --audio-dir /workspace/10_hours/source_audio \
    --sample-size 10 \
&& bash tools/preprocess_polish.sh \
    /workspace/10_hours/transcripts \
    /workspace/10_hours/source_audio \
    processed_data_polish \
    checkpoints/polish_bpe.model
```

## Troubleshooting

### Tokenizer produces `<unk>` tokens
```bash
# Retrain with byte-fallback (add --byte-fallback flag)
python tools/tokenizer/train_bpe.py \
    --manifest-dir /workspace/10_hours/transcripts \
    --manifest-format whisperx \
    --language pl \
    --output-prefix checkpoints/polish_bpe_v2 \
    --vocab-size 12000 \
    --character-coverage 0.9995 \
    --byte-fallback
```

### Audio files not found
```bash
# Test audio discovery to diagnose
python tools/test_audio_discovery.py \
    --transcript-dir /workspace/10_hours/transcripts \
    --audio-dir /workspace/10_hours/source_audio \
    --sample-size 10

# Check directory contents
ls /workspace/10_hours/transcripts/*.json | head -5
ls /workspace/10_hours/source_audio/*.wav | head -5
```

### Out of memory during preprocessing
```bash
# Use CPU instead of GPU
python tools/preprocess_polish.py \
    --data-dir /workspace/10_hours/transcripts \
    --audio-dir /workspace/10_hours/source_audio \
    --device cpu \
    --max-duration 15.0
```

### Test on small subset first
```bash
# Process only 100 samples for testing
python tools/preprocess_polish.py \
    --data-dir /workspace/10_hours/transcripts \
    --audio-dir /workspace/10_hours/source_audio \
    --output-dir processed_data_polish_test \
    --max-samples 100
```

## Key Files

| File | Purpose |
|------|---------|
| `checkpoints/polish_bpe.model` | Trained tokenizer |
| `checkpoints/polish_bpe.vocab` | Vocabulary (human-readable) |
| `processed_data_polish/train_manifest.jsonl` | Training data manifest |
| `processed_data_polish/val_manifest.jsonl` | Validation data manifest |
| `processed_data_polish/stats.json` | Dataset statistics |
| `processed_data_polish/codes/` | Semantic codes (.npy files) |
| `processed_data_polish/condition/` | Conditioning latents |
| `processed_data_polish/emo_vec/` | Emotion vectors |
| `processed_data_polish/text_ids/` | Tokenized text |

## Default Parameters

| Parameter | Value | Why |
|-----------|-------|-----|
| Vocab size | 12000 | Good for Polish morphology |
| Character coverage | 0.9995 | Captures all Polish diacritics |
| Byte fallback | Enabled | Prevents `<unk>` tokens |
| Min duration | 0.5s | Filter very short clips |
| Max duration | 30.0s | Prevent memory issues |
| Validation ratio | 0.01 | 1% for validation |

## Next Steps After Preprocessing

1. ✅ Check `processed_data_polish/stats.json`
2. ✅ Verify manifests have data
3. ✅ Inspect a few `.npy` files
4. 🚀 Start TTS training!

## Documentation

- **Tokenizer**: [tools/tokenizer/README_POLISH.md](tokenizer/README_POLISH.md)
- **Preprocessing**: [tools/README_POLISH_PREPROCESSING.md](README_POLISH_PREPROCESSING.md)
- **Complete Guide**: [tools/POLISH_SETUP_GUIDE.md](POLISH_SETUP_GUIDE.md)
