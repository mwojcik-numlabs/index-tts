#!/bin/bash
# Preprocess Polish WhisperX dataset for IndexTTS2 training
#
# Usage:
#   bash tools/preprocess_polish.sh [transcript_dir] [audio_dir] [output_dir] [tokenizer_path]
#
# Example:
#   bash tools/preprocess_polish.sh \
#       /workspace/10_hours/transcripts \
#       /workspace/10_hours/source_audio \
#       processed_data_polish \
#       checkpoints/polish_bpe.model

# Default values
DATA_DIR="${1:-/workspace/10_hours/transcripts}"
AUDIO_DIR="${2:-/workspace/10_hours/source_audio}"
OUTPUT_DIR="${3:-processed_data_polish}"
TOKENIZER="${4:-checkpoints/polish_bpe.model}"
CONFIG="${5:-checkpoints/config.yaml}"
GPT_CHECKPOINT="${6:-checkpoints/gpt.pth}"

echo "================================================================================"
echo "Preprocessing Polish WhisperX Dataset"
echo "================================================================================"
echo "Transcript directory: ${DATA_DIR}"
echo "Audio directory:      ${AUDIO_DIR}"
echo "Output directory:     ${OUTPUT_DIR}"
echo "Tokenizer:            ${TOKENIZER}"
echo "Config:               ${CONFIG}"
echo "GPT checkpoint:       ${GPT_CHECKPOINT}"
echo "================================================================================"
echo ""

# Check if required files exist
if [ ! -f "${TOKENIZER}" ]; then
    echo "❌ Error: Tokenizer not found at ${TOKENIZER}"
    echo "   Please train the tokenizer first:"
    echo "   bash tools/tokenizer/train_polish_bpe.sh"
    exit 1
fi

if [ ! -f "${CONFIG}" ]; then
    echo "❌ Error: Config file not found at ${CONFIG}"
    exit 1
fi

if [ ! -f "${GPT_CHECKPOINT}" ]; then
    echo "❌ Error: GPT checkpoint not found at ${GPT_CHECKPOINT}"
    echo "   Please download the base checkpoint first."
    exit 1
fi

if [ ! -d "${DATA_DIR}" ]; then
    echo "❌ Error: Transcript directory not found at ${DATA_DIR}"
    exit 1
fi

if [ ! -d "${AUDIO_DIR}" ]; then
    echo "❌ Error: Audio directory not found at ${AUDIO_DIR}"
    exit 1
fi

echo "✓ All required files and directories found"
echo ""

# Run preprocessing
uv run tools/preprocess_polish.py \
    --data-dir "${DATA_DIR}" \
    --audio-dir "${AUDIO_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --tokenizer "${TOKENIZER}" \
    --config "${CONFIG}" \
    --gpt-checkpoint "${GPT_CHECKPOINT}" \
    --device cuda \
    --val-ratio 0.01 \
    --min-duration 0.5 \
    --max-duration 30.0 \
    --skip-existing

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "================================================================================"
    echo "✓ Preprocessing complete!"
    echo "================================================================================"
    echo "Output directory: ${OUTPUT_DIR}"
    echo "  - train_manifest.jsonl: Training samples"
    echo "  - val_manifest.jsonl:   Validation samples"
    echo "  - stats.json:           Dataset statistics"
    echo "  - codes/:               Semantic codes"
    echo "  - condition/:           Conditioning latents"
    echo "  - emo_vec/:             Emotion vectors"
    echo "  - text_ids/:            Tokenized text"
    echo "================================================================================"
else
    echo ""
    echo "❌ Preprocessing failed with exit code ${EXIT_CODE}"
    exit $EXIT_CODE
fi
