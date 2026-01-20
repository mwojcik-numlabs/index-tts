# Complete Polish TTS Training Setup Guide

This guide covers the full pipeline from WhisperX transcripts to trained Polish TTS model.

## Overview

```
WhisperX Transcripts → Tokenizer Training → Dataset Preprocessing → TTS Training
```

## Step 1: Prepare Your Data

Ensure your data is organized:

```
/workspace/10_hours/
├── transcripts/
│   ├── file_001.json  # WhisperX format with utterances
│   ├── file_001.wav   # Corresponding audio
│   ├── file_002.json
│   ├── file_002.wav
│   └── ...
```

### WhisperX JSON Format

```json
{
    "whisperx": {
        "utterances": [
            {
                "start": 0.0,
                "end": 5.5,
                "text": "Dzień dobry, jak się masz?",
                "speaker": null,
                "score": 0.95
            }
        ]
    }
}
```

## Step 2: Train BPE Tokenizer

Train a Polish-specific tokenizer on your transcripts.

### Quick Method

```bash
bash tools/tokenizer/train_polish_bpe.sh \
    /workspace/10_hours/transcripts \
    checkpoints/polish_bpe \
    12000
```

### Detailed Method (with byte-fallback for no `<unk>`)

```bash
python tools/tokenizer/train_bpe.py \
    --manifest-dir /workspace/10_hours/transcripts \
    --manifest-format whisperx \
    --language pl \
    --output-prefix checkpoints/polish_bpe \
    --vocab-size 12000 \
    --character-coverage 0.9995 \
    --byte-fallback
```

**Output:**
- `checkpoints/polish_bpe.model` (tokenizer model)
- `checkpoints/polish_bpe.vocab` (vocabulary)

**Key Parameters:**
- `--vocab-size 12000`: Good balance for Polish
- `--character-coverage 0.9995`: Captures Polish diacritics (ą, ć, ę, ł, ń, ó, ś, ź, ż)
- `--byte-fallback`: Prevents `<unk>` tokens (recommended!)

### Verify Tokenizer

```bash
# Test on sample text
python tools/tokenizer/verify_tokenizer.py \
    --tokenizer checkpoints/polish_bpe.model \
    --language pl \
    --text "wątpię nie było żadnych znaków"

# Test on dataset
python tools/tokenizer/verify_tokenizer.py \
    --tokenizer checkpoints/polish_bpe.model \
    --test-dir /workspace/10_hours/transcripts \
    --language pl \
    --sample-size 1000 \
    --show-examples
```

**Expected:** `✓ SUCCESS: No <unk> tokens found!`

### Analyze Tokenizer Coverage

```bash
python tools/tokenizer/analyze_tokenizer.py \
    --tokenizer checkpoints/polish_bpe.model \
    --language pl
```

## Step 3: Download Base Checkpoints

Download the IndexTTS2 base model:

```bash
# Example (adjust based on actual model location)
mkdir -p checkpoints
cd checkpoints

# Download base GPT checkpoint
wget https://[model-url]/gpt.pth

# Download config
wget https://[model-url]/config.yaml

# Download wav2vec2bert stats
wget https://[model-url]/wav2vec2bert_stats.pt

cd ..
```

## Step 4: Preprocess Dataset

Convert WhisperX transcripts to training-ready features.

### Quick Method

```bash
bash tools/preprocess_polish.sh \
    /workspace/10_hours/transcripts \
    processed_data_polish \
    checkpoints/polish_bpe.model
```

### Detailed Method

```bash
python tools/preprocess_polish.py \
    --data-dir /workspace/10_hours/transcripts \
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

**Output:**
- `processed_data_polish/train_manifest.jsonl` (training samples)
- `processed_data_polish/val_manifest.jsonl` (validation samples)
- `processed_data_polish/stats.json` (dataset statistics)
- `processed_data_polish/codes/` (semantic codes)
- `processed_data_polish/condition/` (conditioning latents)
- `processed_data_polish/emo_vec/` (emotion vectors)
- `processed_data_polish/text_ids/` (tokenized text)

### Check Preprocessing Output

```bash
# View statistics
cat processed_data_polish/stats.json

# View sample manifest entries
head -n 5 processed_data_polish/train_manifest.jsonl

# Count samples
wc -l processed_data_polish/train_manifest.jsonl
wc -l processed_data_polish/val_manifest.jsonl
```

## Step 5: Train TTS Model

Configure and train your Polish TTS model.

### Update Config for Polish

Edit your training config to use Polish-specific settings:

```yaml
# training_config_polish.yaml
tokenizer:
  path: checkpoints/polish_bpe.model
  language: pl

data:
  train_manifest: processed_data_polish/train_manifest.jsonl
  val_manifest: processed_data_polish/val_manifest.jsonl

training:
  batch_size: 8
  learning_rate: 1e-5
  max_steps: 100000
  warmup_steps: 5000

checkpoint:
  base_model: checkpoints/gpt.pth
  save_dir: checkpoints_polish
```

### Start Training

```bash
python train.py \
    --config training_config_polish.yaml \
    --output-dir checkpoints_polish \
    --device cuda
```

## Complete Pipeline Example

Here's the complete workflow from start to finish:

```bash
# Set up paths
export DATA_DIR="/workspace/10_hours/transcripts"
export OUTPUT_DIR="processed_data_polish"
export CHECKPOINT_DIR="checkpoints"
export TOKENIZER="${CHECKPOINT_DIR}/polish_bpe.model"

# Step 1: Train tokenizer
echo "Step 1/4: Training tokenizer..."
python tools/tokenizer/train_bpe.py \
    --manifest-dir "${DATA_DIR}" \
    --manifest-format whisperx \
    --language pl \
    --output-prefix "${CHECKPOINT_DIR}/polish_bpe" \
    --vocab-size 12000 \
    --character-coverage 0.9995 \
    --byte-fallback

# Step 2: Verify tokenizer
echo "Step 2/4: Verifying tokenizer..."
python tools/tokenizer/verify_tokenizer.py \
    --tokenizer "${TOKENIZER}" \
    --test-dir "${DATA_DIR}" \
    --language pl \
    --sample-size 1000

# Step 3: Preprocess dataset
echo "Step 3/4: Preprocessing dataset..."
python tools/preprocess_polish.py \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --tokenizer "${TOKENIZER}" \
    --config "${CHECKPOINT_DIR}/config.yaml" \
    --gpt-checkpoint "${CHECKPOINT_DIR}/gpt.pth" \
    --device cuda \
    --val-ratio 0.01 \
    --skip-existing

# Step 4: Check output
echo "Step 4/4: Checking output..."
cat "${OUTPUT_DIR}/stats.json"
echo ""
echo "✓ Setup complete! Ready for training."
echo "Training manifest: ${OUTPUT_DIR}/train_manifest.jsonl"
echo "Validation manifest: ${OUTPUT_DIR}/val_manifest.jsonl"
```

## Key Differences from Japanese Pipeline

| Aspect | Japanese | Polish |
|--------|----------|--------|
| Input format | JSONL with `text` field | WhisperX JSON with nested `utterances` |
| Language code | `ja` | `pl` |
| Text normalization | Japanese-specific (hiragana, katakana) | Generic/basic cleanup |
| Character coverage | 0.9995 (for kanji/kana) | 0.9995 (for diacritics) |
| Vocabulary size | 12000 (default) | 12000 (recommended) |
| Byte fallback | Optional | **Recommended** |
| Audio segmentation | Full file | Utterance-level (start/end times) |

## Recommended Settings

### For Polish Language

| Parameter | Value | Reason |
|-----------|-------|--------|
| Vocabulary size | 12000 | Good balance for Polish morphology |
| Character coverage | 0.9995 | Captures all diacritics (ą, ć, ę, ł, ń, ó, ś, ź, ż) |
| Byte fallback | Enabled | Prevents `<unk>` tokens |
| Min duration | 0.5-1.0s | Filter out very short clips |
| Max duration | 15-30s | Prevent memory issues |
| Validation ratio | 0.01-0.02 | 1-2% for validation |

### Hardware Requirements

- **Tokenizer training**: CPU sufficient, <1GB RAM
- **Dataset preprocessing**:
  - GPU: CUDA-capable (recommended)
  - RAM: 16GB+ recommended
  - Disk: ~10GB per 10 hours of audio
- **TTS training**:
  - GPU: 24GB+ VRAM (e.g., RTX 3090, A100)
  - RAM: 32GB+ recommended

## Troubleshooting

### Issue: Tokenizer creates `<unk>` tokens

**Solution:** Retrain with `--byte-fallback`:
```bash
python tools/tokenizer/train_bpe.py \
    --manifest-dir /workspace/10_hours/transcripts \
    --manifest-format whisperx \
    --language pl \
    --output-prefix checkpoints/polish_bpe_v2 \
    --vocab-size 12000 \
    --character-coverage 0.9995 \
    --byte-fallback
```

### Issue: Audio files not found during preprocessing

**Solution:** Specify audio directory separately:
```bash
python tools/preprocess_polish.py \
    --data-dir /workspace/10_hours/transcripts \
    --audio-dir /workspace/10_hours/audio \
    --output-dir processed_data_polish
```

### Issue: Out of memory during preprocessing

**Solution:** Use CPU or reduce max duration:
```bash
python tools/preprocess_polish.py \
    --data-dir /workspace/10_hours/transcripts \
    --device cpu \
    --max-duration 15.0 \
    --output-dir processed_data_polish
```

### Issue: Empty manifests after preprocessing

**Check:**
1. Audio files are discoverable
2. Duration constraints aren't too strict
3. WhisperX JSON format is correct
4. Tokenizer is working properly

## File Structure Summary

```
index-tts2-training/
├── checkpoints/
│   ├── polish_bpe.model         # Trained tokenizer
│   ├── polish_bpe.vocab         # Vocabulary
│   ├── config.yaml              # Base config
│   ├── gpt.pth                  # Base checkpoint
│   └── wav2vec2bert_stats.pt    # W2V stats
├── processed_data_polish/
│   ├── train_manifest.jsonl     # Training data
│   ├── val_manifest.jsonl       # Validation data
│   ├── stats.json               # Statistics
│   ├── codes/                   # Semantic codes
│   ├── condition/               # Conditioning
│   ├── emo_vec/                 # Emotion vectors
│   └── text_ids/                # Tokenized text
└── tools/
    ├── tokenizer/
    │   ├── train_bpe.py         # Tokenizer training
    │   ├── verify_tokenizer.py  # Verification
    │   ├── analyze_tokenizer.py # Analysis
    │   └── README_POLISH.md     # Tokenizer docs
    ├── preprocess_polish.py     # Preprocessing script
    ├── preprocess_polish.sh     # Helper script
    └── README_POLISH_PREPROCESSING.md  # Preprocessing docs
```

## Next Steps

1. ✅ Complete steps 1-4 above
2. 🚀 Start TTS training with preprocessed data
3. 📊 Monitor training metrics
4. 🎯 Evaluate on validation set
5. 🎤 Generate Polish speech samples!

## Documentation References

- **Tokenizer Training**: [tools/tokenizer/README_POLISH.md](tokenizer/README_POLISH.md)
- **Tokenizer Quick Start**: [tools/tokenizer/QUICK_START.md](tokenizer/QUICK_START.md)
- **Preprocessing**: [tools/README_POLISH_PREPROCESSING.md](README_POLISH_PREPROCESSING.md)
- **Original Japanese Script**: [tools/preprocess_japanese.py](preprocess_japanese.py)

## Support

If you encounter issues:
1. Check the troubleshooting sections in the documentation
2. Verify all prerequisites are met
3. Test with a small subset first (`--max-samples 100`)
4. Review error messages carefully - they often indicate the issue
