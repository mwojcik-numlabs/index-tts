# Polish WhisperX Dataset Preprocessing for IndexTTS2

This guide explains how to preprocess your Polish WhisperX transcripts for IndexTTS2 fine-tuning.

## Overview

The preprocessing script [preprocess_polish.py](preprocess_polish.py) converts WhisperX JSON transcripts into training-ready features:

1. **Text Processing**: Normalizes and tokenizes Polish text using your trained BPE tokenizer
2. **Audio Processing**: Loads audio files and extracts utterance segments
3. **Semantic Features**: Extracts semantic features using SeamlessM4T + Wav2Vec2Bert
4. **Semantic Codes**: Quantizes features with RepCodec
5. **Conditioning**: Extracts conditioning latents and emotion vectors with UnifiedVoice v2
6. **Manifests**: Generates train/validation JSONL manifests

## Prerequisites

Before preprocessing, ensure you have:

1. ✅ **Trained Polish BPE tokenizer** (see [tokenizer/README_POLISH.md](tokenizer/README_POLISH.md))
2. ✅ **IndexTTS2 base checkpoint** (`checkpoints/gpt.pth`)
3. ✅ **Config file** (`checkpoints/config.yaml`)
4. ✅ **WhisperX transcripts** (JSON files with utterances)
5. ✅ **Audio files** (WAV format, matching transcripts)

## Quick Start

### Using the Helper Script (Easiest)

```bash
bash tools/preprocess_polish.sh \
    /workspace/10_hours/transcripts \
    /workspace/10_hours/source_audio \
    processed_data_polish \
    checkpoints/polish_bpe.model
```

This will:
- Process all JSON transcripts in `/workspace/10_hours/transcripts`
- Find corresponding audio files in `/workspace/10_hours/source_audio`
- Save features to `processed_data_polish/`
- Use Polish tokenizer at `checkpoints/polish_bpe.model`

### Using Python Directly (More Control)

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

## Command-Line Arguments

### Required Arguments

| Argument | Description |
|----------|-------------|
| `--data-dir` | Root directory containing WhisperX JSON transcripts |
| `--tokenizer` | Path to trained Polish BPE tokenizer (`.model` file) |
| `--config` | IndexTTS config YAML file |
| `--gpt-checkpoint` | UnifiedVoice checkpoint for conditioning extraction |

### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--audio-dir` | Same as `--data-dir` | Directory with audio files (if separate) |
| `--output-dir` | `processed_data_polish` | Output directory for features |
| `--device` | `cuda` | Computation device (`cuda` or `cpu`) |
| `--val-ratio` | `0.01` | Validation split ratio (1% default) |
| `--seed` | `17` | Random seed for reproducibility |
| `--max-samples` | `0` | Limit samples for testing (0 = all) |
| `--audio-sr` | `24000` | Target audio sampling rate |
| `--skip-existing` | `False` | Skip already processed samples |
| `--audio-extension` | `.wav` | Audio file extension |
| `--min-duration` | `0.5` | Minimum utterance duration (seconds) |
| `--max-duration` | `30.0` | Maximum utterance duration (seconds) |

## Directory Structure

### Input Structure

Your WhisperX data should be organized like this:

```
/workspace/10_hours/
├── transcripts/            # JSON transcripts
│   ├── file_001.json
│   ├── file_002.json
│   └── ...
└── source_audio/           # Corresponding audio files
    ├── file_001.wav
    ├── file_002.wav
    └── ...
```

**Alternative structures also supported:**

```
# Files side-by-side
/workspace/10_hours/transcripts/
├── file_001.json
├── file_001.wav
├── file_002.json
├── file_002.wav
└── ...

# Or with 'audio' directory name
/workspace/10_hours/
├── transcripts/
│   └── *.json
└── audio/
    └── *.wav
```

### Output Structure

After preprocessing, you'll have:

```
processed_data_polish/
├── train_manifest.jsonl    # Training samples
├── val_manifest.jsonl      # Validation samples
├── stats.json              # Dataset statistics
├── codes/                  # Semantic codes (.npy)
│   ├── audio_001_0000.npy
│   ├── audio_001_0001.npy
│   └── ...
├── condition/              # Conditioning latents (.npy)
│   ├── audio_001_0000.npy
│   └── ...
├── emo_vec/                # Emotion vectors (.npy)
│   ├── audio_001_0000.npy
│   └── ...
└── text_ids/               # Tokenized text (.npy)
    ├── audio_001_0000.npy
    └── ...
```

## WhisperX Format

Your JSON transcripts should follow this format:

```json
{
    "whisperx": {
        "utterances": [
            {
                "start": 0.031,
                "end": 14.513,
                "text": "wątpię nie było żadnych znaków...",
                "score": 0.9124615384615384,
                "speaker": null,
                "emotions": null
            }
        ]
    }
}
```

The script extracts:
- `text`: Utterance text (required)
- `start`, `end`: Time boundaries for audio segmentation
- `speaker`: Speaker identifier (optional)
- `score`: Confidence score (optional)

## Audio File Discovery

The script automatically finds audio files using multiple strategies:

1. **Explicit audio directory** (recommended): `/workspace/10_hours/source_audio/file.wav`
2. **Same directory**: `transcripts/file.json` → `transcripts/file.wav`
3. **Sibling directories**: Automatically checks `source_audio/`, `audio/`, `wav/`, `wavs/`
4. **Nested structures**: Preserves subdirectory structure when present

### Test Audio Discovery First

Before running full preprocessing, verify audio files can be found:

```bash
python tools/test_audio_discovery.py \
    --transcript-dir /workspace/10_hours/transcripts \
    --audio-dir /workspace/10_hours/source_audio \
    --sample-size 10
```

**Expected output:**
```
✓ All audio files found! You can proceed with preprocessing:

  bash tools/preprocess_polish.sh \
      /workspace/10_hours/transcripts \
      /workspace/10_hours/source_audio \
      processed_data_polish \
      checkpoints/polish_bpe.model
```

### Specify Audio Directory

For the structure `/workspace/10_hours/transcripts/` + `/workspace/10_hours/source_audio/`:

```bash
python tools/preprocess_polish.py \
    --data-dir /workspace/10_hours/transcripts \
    --audio-dir /workspace/10_hours/source_audio \
    --output-dir processed_data_polish
```

If audio files are in a sibling directory named `source_audio`, `audio`, or `wav`, the script will find them automatically even without `--audio-dir`.

## Duration Filtering

Control which utterances to include based on duration:

```bash
python tools/preprocess_polish.py \
    --data-dir /workspace/10_hours/transcripts \
    --min-duration 1.0 \    # Skip utterances < 1 second
    --max-duration 15.0 \   # Skip utterances > 15 seconds
    --output-dir processed_data_polish
```

**Recommendations:**
- **Min duration**: 0.5-1.0 seconds (avoid very short clips)
- **Max duration**: 15-30 seconds (avoid memory issues)

## Incremental Processing

Resume preprocessing after interruption:

```bash
python tools/preprocess_polish.py \
    --data-dir /workspace/10_hours/transcripts \
    --output-dir processed_data_polish \
    --skip-existing  # Skip already processed samples
```

The script tracks processed samples in the manifests and `.npy` files, allowing safe resumption.

## Output Manifests

### Training Manifest (`train_manifest.jsonl`)

Each line is a JSON object:

```json
{
    "id": "audio_001_0000",
    "audio_path": "/workspace/10_hours/transcripts/audio_001.wav",
    "text": "wątpię nie było żadnych znaków",
    "speaker": "speaker_001",
    "language": "pl",
    "duration": 14.482,
    "start": 0.031,
    "end": 14.513,
    "text_ids_path": "processed_data_polish/text_ids/audio_001_0000.npy",
    "text_len": 45,
    "codes_path": "processed_data_polish/codes/audio_001_0000.npy",
    "code_len": 362,
    "condition_path": "processed_data_polish/condition/audio_001_0000.npy",
    "condition_len": 1024,
    "emo_vec_path": "processed_data_polish/emo_vec/audio_001_0000.npy"
}
```

### Statistics (`stats.json`)

Summary of preprocessing:

```json
{
    "total": 10543,
    "train": 10438,
    "val": 105,
    "tokenizer": "checkpoints/polish_bpe.model",
    "gpt_checkpoint": "checkpoints/gpt.pth"
}
```

## Validation Split

The script uses deterministic hashing to assign samples to train/validation:

- Consistent across runs (same sample always goes to same split)
- Controlled by `--val-ratio` (default: 0.01 = 1%)
- Reproducible with `--seed`

```bash
# Use 2% for validation
python tools/preprocess_polish.py \
    --data-dir /workspace/10_hours/transcripts \
    --val-ratio 0.02
```

## Performance Tips

1. **GPU Usage**: Enable CUDA for faster processing
   ```bash
   --device cuda
   ```

2. **Skip Existing**: Resume after interruption
   ```bash
   --skip-existing
   ```

3. **Limit Samples**: Test on subset first
   ```bash
   --max-samples 100  # Process only 100 samples
   ```

4. **Batch Processing**: Process in chunks if memory is limited
   ```bash
   # Process first 1000, then next 1000, etc.
   --max-samples 1000
   ```

## Troubleshooting

### Problem: "No audio file found"

**Cause**: Audio file naming doesn't match transcript

**Solution**:
1. Check audio file extension: `--audio-extension .wav`
2. Specify audio directory: `--audio-dir /path/to/audio`
3. Verify file names match (e.g., `audio_001.json` → `audio_001.wav`)

### Problem: "Failed to load audio"

**Cause**: Corrupt audio file or unsupported format

**Solution**:
1. Verify audio files are valid WAV format
2. Check file permissions
3. Try converting to standard format:
   ```bash
   ffmpeg -i input.mp3 -ar 24000 -ac 1 output.wav
   ```

### Problem: Out of memory

**Cause**: Processing very long utterances

**Solution**:
1. Reduce `--max-duration`:
   ```bash
   --max-duration 15.0  # Limit to 15 seconds
   ```
2. Use CPU instead of GPU:
   ```bash
   --device cpu
   ```

### Problem: Text tokenization fails

**Cause**: Tokenizer produces `<unk>` tokens

**Solution**:
1. Verify tokenizer was trained with `--byte-fallback`
2. Check tokenizer quality:
   ```bash
   python tools/tokenizer/verify_tokenizer.py \
       --tokenizer checkpoints/polish_bpe.model \
       --test-dir /workspace/10_hours/transcripts \
       --language pl
   ```
3. Retrain tokenizer if needed (see [tokenizer/README_POLISH.md](tokenizer/README_POLISH.md))

### Problem: Empty manifests

**Cause**: All samples filtered out or audio files not found

**Solution**:
1. Check duration constraints (`--min-duration`, `--max-duration`)
2. Verify audio files exist and are discoverable
3. Check transcript format is correct (WhisperX JSON)
4. Run without `--skip-existing` to reprocess

## Example Workflows

### Basic Workflow

```bash
# 1. Train tokenizer
bash tools/tokenizer/train_polish_bpe.sh \
    /workspace/10_hours/transcripts \
    checkpoints/polish_bpe \
    12000

# 2. Verify tokenizer
python tools/tokenizer/verify_tokenizer.py \
    --tokenizer checkpoints/polish_bpe.model \
    --test-dir /workspace/10_hours/transcripts \
    --language pl \
    --sample-size 1000

# 3. Preprocess dataset
bash tools/preprocess_polish.sh \
    /workspace/10_hours/transcripts \
    processed_data_polish \
    checkpoints/polish_bpe.model

# 4. Check output
cat processed_data_polish/stats.json
head -n 5 processed_data_polish/train_manifest.jsonl
```

### Advanced Workflow with Filtering

```bash
python tools/preprocess_polish.py \
    --data-dir /workspace/10_hours/transcripts \
    --audio-dir /workspace/10_hours/audio \
    --output-dir processed_data_polish \
    --tokenizer checkpoints/polish_bpe.model \
    --config checkpoints/config.yaml \
    --gpt-checkpoint checkpoints/gpt.pth \
    --device cuda \
    --val-ratio 0.02 \
    --min-duration 1.0 \
    --max-duration 20.0 \
    --skip-existing \
    --audio-extension .wav
```

## Next Steps

After preprocessing, you need to create prompt-target pairs for GPT training:

### Step 1: Build Prompt-Target Pairs

The GPT trainer requires paired manifests (prompt + target). Build them:

```bash
# Training pairs
python tools/build_gpt_prompt_pairs.py \
    --manifest processed_data_polish/train_manifest.jsonl \
    --output processed_data_polish/train_pairs.jsonl \
    --pairs-per-target 2 \
    --seed 2025

# Validation pairs
python tools/build_gpt_prompt_pairs.py \
    --manifest processed_data_polish/val_manifest.jsonl \
    --output processed_data_polish/val_pairs.jsonl \
    --pairs-per-target 2 \
    --seed 2025
```

**What this does:**
- Groups utterances by speaker
- Creates prompt-target pairs from the same speaker
- Each target gets 2 different prompts (data augmentation)
- Generates: `processed_data_polish/train_pairs.jsonl` and `val_pairs.jsonl`

### Step 2: Verify Paired Manifests

```bash
# Check pair counts
wc -l processed_data_polish/train_pairs.jsonl
wc -l processed_data_polish/val_pairs.jsonl

# View sample pair
head -n 1 processed_data_polish/train_pairs.jsonl | jq .
```

### Step 3: Start Training

```bash
python trainers/train_gpt_v2.py \
    --train-manifest processed_data_polish/train_pairs.jsonl::pl \
    --val-manifest processed_data_polish/val_pairs.jsonl::pl \
    --tokenizer checkpoints/polish_bpe.model \
    --config checkpoints/config.yaml \
    --base-checkpoint checkpoints/gpt.pth \
    --output-dir trained_ckpts_polish \
    --batch-size 4 \
    --grad-accumulation 4 \
    --epochs 10 \
    --learning-rate 2e-5 \
    --warmup-steps 1000 \
    --amp
```

Note: The `::pl` suffix indicates Polish language for the dataset.

## Additional Resources

- [Polish Tokenizer Training](tokenizer/README_POLISH.md)
- [Polish Tokenizer Quick Start](tokenizer/QUICK_START.md)
- Original Japanese preprocessing: [preprocess_japanese.py](preprocess_japanese.py)
