# Polish BPE Tokenizer - Quick Start Guide

## Step 1: Train the Tokenizer

### Option A: Using the Helper Script (Easiest)

```bash
bash tools/tokenizer/train_polish_bpe.sh /workspace/10_hours/transcripts checkpoints/polish_bpe 12000
```

### Option B: Direct Python Command (More Control)

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

**Note:** Adding `--byte-fallback` prevents `<unk>` tokens (recommended for production).

---

## Step 2: Verify No `<unk>` Tokens

### Quick Single Text Test

```bash
python tools/tokenizer/verify_tokenizer.py \
    --tokenizer checkpoints/polish_bpe.model \
    --language pl \
    --text "wątpię nie było żadnych znaków"
```

### Comprehensive Dataset Test

```bash
python tools/tokenizer/verify_tokenizer.py \
    --tokenizer checkpoints/polish_bpe.model \
    --test-dir /workspace/10_hours/transcripts \
    --language pl \
    --sample-size 1000 \
    --show-examples
```

**Expected output:** `✓ SUCCESS: No <unk> tokens found!`

---

## Step 3: Use the Tokenizer

```python
from indextts.utils.front import TextNormalizer, TextTokenizer

# Load
normalizer = TextNormalizer(preferred_language="pl")
tokenizer = TextTokenizer(
    vocab_file="checkpoints/polish_bpe.model",
    normalizer=normalizer
)

# Tokenize
text = "Dzień dobry, jak się masz?"
tokens = tokenizer.tokenize(text, language="pl")
ids = tokenizer.encode(text, language="pl")

# Verify no <unk>
assert tokenizer.unk_token_id not in ids, "Found <unk> token!"

# Decode
decoded = tokenizer.decode(ids)
```

---

## Troubleshooting

### Problem: Found `<unk>` tokens

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

### Problem: "No JSON files found"

**Check:**
- Path is correct: `ls /workspace/10_hours/transcripts/*.json`
- Files exist in subdirectories (script searches recursively)

### Problem: Training is too slow

**Solutions:**
- Reduce `--sample-size`: Use fewer samples for faster training
- Use `--input-sentence-size 1000000` to limit training data

---

## Key Parameters

| Parameter | Recommended Value | Notes |
|-----------|------------------|-------|
| `--vocab-size` | 12000 | Good balance; try 8000-15000 |
| `--character-coverage` | 0.9995 | Captures Polish diacritics |
| `--byte-fallback` | `--byte-fallback` | Prevents `<unk>` tokens |
| `--language` | `pl` | Polish language code |

---

## Files Produced

After training, you'll have:
- `checkpoints/polish_bpe.model` - SentencePiece model (use this for tokenization)
- `checkpoints/polish_bpe.vocab` - Vocabulary file (human-readable)

---

## Next Steps

1. ✅ Train tokenizer
2. ✅ Verify no `<unk>` tokens
3. ✅ Integrate into your TTS training pipeline
4. 🎉 Profit!

For detailed documentation, see [README_POLISH.md](README_POLISH.md)
