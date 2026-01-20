# Training a Polish BPE Tokenizer

This guide explains how to train a BPE tokenizer for Polish language using WhisperX transcript data.

## Overview

The `train_bpe.py` script has been enhanced to support multiple data formats:

1. **JSONL format** (original): Files with one JSON object per line, each containing a `text` field
2. **WhisperX format** (new): Nested JSON files with `whisperx.utterances[].text` structure

## Quick Start

### Using the Helper Script

The easiest way to train a Polish tokenizer:

```bash
bash tools/tokenizer/train_polish_bpe.sh /workspace/10_hours/transcripts checkpoints/polish_bpe 12000
```

Parameters:
- `$1`: Directory containing WhisperX JSON files (default: `/workspace/10_hours/transcripts`)
- `$2`: Output prefix for tokenizer files (default: `checkpoints/polish_bpe`)
- `$3`: Vocabulary size (default: `12000`)

### Using the Python Script Directly

For more control, use the Python script directly:

```bash
python tools/tokenizer/train_bpe.py \
    --manifest-dir /workspace/10_hours/transcripts \
    --manifest-format whisperx \
    --language pl \
    --output-prefix checkpoints/polish_bpe \
    --vocab-size 12000 \
    --character-coverage 0.9995 \
    --model-type bpe
```

## Command-Line Arguments

### Required Arguments (choose one):

- `--manifest`: One or more JSONL files with `text` field (for JSONL format)
- `--manifest-dir`: Directory containing JSON files (for WhisperX format)

### Format and Language:

- `--manifest-format`: Format type, either `jsonl` (default) or `whisperx`
- `--language`: Language code for text normalization (e.g., `pl`, `ja`, `zh`, `en`). Default: `ja`

### Tokenizer Configuration:

- `--output-prefix`: Output prefix for `.model` and `.vocab` files. Default: `checkpoints/tokenizer_bpe`
- `--vocab-size`: Vocabulary size. Default: `12000`
- `--character-coverage`: Character coverage for SentencePiece (0.0-1.0). Default: `0.9995`
  - Keep near 1.0 for non-Latin languages to include rare characters
  - For Polish (Latin-based), you could use 0.995-0.9995
- `--model-type`: Model type, either `bpe` or `unigram`. Default: `bpe`
- `--input-sentence-size`: Limit training sentences (0 = use all). Default: `0`
- `--byte-fallback`: Enable byte fallback to avoid `<unk>` tokens. Default: `False`

## Polish Language Considerations

### Character Coverage

Polish uses Latin script with diacritics (ą, ć, ę, ł, ń, ó, ś, ź, ż). Recommended settings:

- `--character-coverage 0.9995`: Captures most Polish characters including rare diacritics
- `--character-coverage 0.995`: Faster training, may miss very rare characters

### Vocabulary Size

Recommended vocabulary sizes for Polish:

- **8,000-10,000**: Sufficient for most TTS tasks
- **12,000**: Good balance (recommended)
- **15,000-20,000**: Better for complex applications with technical vocabulary

### Text Normalization

The script uses `TextNormalizer` with language-specific rules:

- For Polish (`--language pl`): Uses basic cleanup (punctuation normalization, whitespace handling)
- Currently, the normalizer doesn't have Polish-specific rules (like it does for Japanese/Chinese)
- Polish text will be processed with the generic normalization path

## Example WhisperX Data Format

Your Polish transcripts should follow this structure:

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
        ],
        "words": [
            {
                "start": 0.031,
                "end": 0.974,
                "text": "wątpię",
                "score": 0.897,
                "speaker": null,
                "emotions": null
            }
        ]
    }
}
```

The script extracts text from `whisperx.utterances[].text` fields.

## Output Files

After training, you'll get two files:

- `{output_prefix}.model`: SentencePiece model file (used for tokenization)
- `{output_prefix}.vocab`: Vocabulary file (human-readable token list)

Example:
```
checkpoints/polish_bpe.model
checkpoints/polish_bpe.vocab
```

## Verifying Your Tokenizer (Checking for `<unk>` tokens)

After training, it's crucial to verify that your tokenizer doesn't produce `<unk>` (unknown) tokens on your dataset.

### Quick Verification

Test a single text:

```bash
python tools/tokenizer/verify_tokenizer.py \
    --tokenizer checkpoints/polish_bpe.model \
    --language pl \
    --text "wątpię nie było żadnych znaków"
```

### Comprehensive Dataset Verification

Test against your entire dataset (or a sample):

```bash
# Test 1000 samples from your dataset
python tools/tokenizer/verify_tokenizer.py \
    --tokenizer checkpoints/polish_bpe.model \
    --test-dir /workspace/10_hours/transcripts \
    --language pl \
    --sample-size 1000 \
    --show-examples
```

The script will report:
- Total texts tested
- Number/percentage of texts with `<unk>` tokens
- Examples of problematic texts (with `--show-examples`)

### Expected Output

**Good result (no `<unk>` tokens):**
```
✓ SUCCESS: No <unk> tokens found!
  Your tokenizer handles all characters in the dataset.
```

**Problem (some `<unk>` tokens found):**
```
⚠️  WARNING: 15 texts contain <unk> tokens
  Consider:
    1. Increasing --character-coverage (e.g., 0.9999)
    2. Using --byte-fallback during training
    3. Increasing --vocab-size
```

### Preventing `<unk>` Tokens

If you get `<unk>` tokens, retrain with these adjustments:

1. **Enable byte fallback** (recommended):
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

2. **Increase character coverage**:
   - Try `--character-coverage 0.9999` (captures even rarer characters)

3. **Increase vocabulary size**:
   - Try `--vocab-size 15000` or higher

## Testing Your Tokenizer (Python)

After verification, you can use the tokenizer programmatically:

```python
from indextts.utils.front import TextNormalizer, TextTokenizer

# Initialize
normalizer = TextNormalizer(preferred_language="pl")
tokenizer = TextTokenizer(
    vocab_file="checkpoints/polish_bpe.model",
    normalizer=normalizer
)

# Test
text = "wątpię nie było żadnych znaków"
tokens = tokenizer.tokenize(text, language="pl")
print("Tokens:", tokens)

# Encode to IDs
ids = tokenizer.encode(text, language="pl")
print("Token IDs:", ids)

# Check for <unk>
if tokenizer.unk_token_id in ids:
    print("⚠️ Warning: Text contains <unk> token!")
else:
    print("✓ No <unk> tokens")

# Decode back
decoded = tokenizer.decode(ids)
print("Decoded:", decoded)
```

## Troubleshooting

### No JSON files found

```
FileNotFoundError: No JSON files found in /path/to/dir
```

**Solution**: Ensure the directory path is correct and contains `.json` files.

### Unexpected format warning

```
[Warning] Skipping file.json: unexpected format
```

**Solution**: Check that your JSON files contain `whisperx.utterances` structure. Files not matching this format will be skipped.

### No non-empty samples found

```
RuntimeError: No non-empty samples found. Cannot train tokenizer.
```

**Solution**: Check that your JSON files contain actual text in the utterances. Empty texts are skipped.

## Comparison: JSONL vs WhisperX Format

### JSONL Format (Original)
```bash
python tools/tokenizer/train_bpe.py \
    --manifest data/train.jsonl \
    --manifest-format jsonl \
    --language ja \
    --output-prefix checkpoints/japanese_bpe \
    --vocab-size 12000
```

Data format:
```jsonl
{"text": "これはテストです"}
{"text": "もう一つの例"}
```

### WhisperX Format (New)
```bash
python tools/tokenizer/train_bpe.py \
    --manifest-dir /workspace/10_hours/transcripts \
    --manifest-format whisperx \
    --language pl \
    --output-prefix checkpoints/polish_bpe \
    --vocab-size 12000
```

Data format: See example above in "Example WhisperX Data Format"

## Additional Notes

- The script recursively searches for all `.json` files in the specified directory
- Multiple utterances per file are supported
- Files with parsing errors are skipped with warnings
- Empty or whitespace-only texts are automatically filtered out
- Text normalization is applied before tokenization training

## Further Customization

If you need Polish-specific text normalization (e.g., handling abbreviations, numbers, etc.), you can extend the `TextNormalizer` class in [indextts/utils/front.py](../../indextts/utils/front.py) to add Polish-specific rules similar to the Japanese normalization.
