#!/bin/bash
# Script to train a Polish BPE tokenizer from WhisperX transcripts
#
# Usage:
#   bash tools/tokenizer/train_polish_bpe.sh [transcript_dir] [output_prefix] [vocab_size]
#
# Example:
#   bash tools/tokenizer/train_polish_bpe.sh /workspace/10_hours/transcripts checkpoints/polish_bpe 12000

# Default values
TRANSCRIPT_DIR="${1:-/workspace/10_hours/transcripts}"
OUTPUT_PREFIX="${2:-checkpoints/polish_bpe}"
VOCAB_SIZE="${3:-12000}"

echo "Training Polish BPE tokenizer..."
echo "  Transcript directory: ${TRANSCRIPT_DIR}"
echo "  Output prefix: ${OUTPUT_PREFIX}"
echo "  Vocabulary size: ${VOCAB_SIZE}"
echo ""

python tools/tokenizer/train_bpe.py \
    --manifest-dir "${TRANSCRIPT_DIR}" \
    --manifest-format whisperx \
    --language pl \
    --output-prefix "${OUTPUT_PREFIX}" \
    --vocab-size "${VOCAB_SIZE}" \
    --character-coverage 0.9995 \
    --model-type bpe

echo ""
echo "Done! Tokenizer files saved:"
echo "  Model: ${OUTPUT_PREFIX}.model"
echo "  Vocab: ${OUTPUT_PREFIX}.vocab"
