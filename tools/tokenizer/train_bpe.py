#!/usr/bin/env python3
"""
Utility script to train a SentencePiece BPE tokenizer on text corpus.

Example (JSONL format with 'text' field):
    python tools/tokenizer/train_bpe.py \\
        --manifest JA_yodas_dataset/ja_yodas_train.jsonl \\
        --output-prefix checkpoints/japanese_bpe \\
        --vocab-size 12000

Example (Polish WhisperX format):
    python tools/tokenizer/train_bpe.py \\
        --manifest-dir /workspace/10_hours/transcripts \\
        --manifest-format whisperx \\
        --language pl \\
        --output-prefix checkpoints/polish_bpe \\
        --vocab-size 12000 \\
        --character-coverage 0.9995
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import sentencepiece as spm

from indextts.utils.front import TextNormalizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer with SentencePiece.")
    parser.add_argument(
        "--manifest",
        nargs="+",
        help="One or more JSONL manifests containing a 'text' field. Mutually exclusive with --manifest-dir.",
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        help="Directory containing JSON files (for whisperx format). Mutually exclusive with --manifest.",
    )
    parser.add_argument(
        "--manifest-format",
        choices=["jsonl", "whisperx"],
        default="jsonl",
        help="Format of the manifest files: 'jsonl' (default) or 'whisperx' (nested JSON with whisperx.utterances).",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="ja",
        help="Language code for text normalization (e.g., 'ja', 'zh', 'en', 'pl'). Default: 'ja'",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("checkpoints/tokenizer_bpe"),
        help="Output prefix for the tokenizer files (.model/.vocab).",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=12000,
        help="Desired vocabulary size.",
    )
    parser.add_argument(
        "--character-coverage",
        type=float,
        default=0.9995,
        help="Character coverage for SentencePiece (keep near 1.0 for non-Latin languages).",
    )
    parser.add_argument(
        "--model-type",
        choices=["bpe", "unigram"],
        default="bpe",
        help="SentencePiece model type.",
    )
    parser.add_argument(
        "--input-sentence-size",
        type=int,
        default=0,
        help="Limit the number of sentences sampled for training (0 means use all).",
    )
    parser.add_argument(
        "--byte-fallback",
        action="store_true",
        default=False,
        help="Enable byte fallback to avoid <unk> for unseen characters (Do not enable unless you know what you're doing)",
    )
    args = parser.parse_args()

    # Validate mutual exclusivity
    if args.manifest and args.manifest_dir:
        parser.error("--manifest and --manifest-dir are mutually exclusive. Use one or the other.")
    if not args.manifest and not args.manifest_dir:
        parser.error("Either --manifest or --manifest-dir must be specified.")

    return args


def iter_texts_jsonl(manifests: list[Path], language: str) -> tuple[int, int, Path]:
    """Process JSONL manifests with 'text' field."""
    normalizer = TextNormalizer(preferred_language=language)
    normalizer.load()

    num_samples = 0
    num_empty = 0
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w", encoding="utf-8")
    try:
        with tmp_file as fp:
            for manifest in manifests:
                with open(manifest, "r", encoding="utf-8") as handle:
                    for line in handle:
                        if not line.strip():
                            continue
                        payload = json.loads(line)
                        text = payload.get("text", "")
                        text = normalizer.normalize(text, language=language)
                        if not text:
                            num_empty += 1
                            continue
                        fp.write(text + "\n")
                        num_samples += 1
    except Exception:
        os.unlink(tmp_file.name)
        raise
    return num_samples, num_empty, Path(tmp_file.name)


def iter_texts_whisperx(manifest_dir: Path, language: str) -> tuple[int, int, Path]:
    """Process WhisperX JSON files with nested utterances."""
    normalizer = TextNormalizer(preferred_language=language)
    normalizer.load()

    num_samples = 0
    num_empty = 0
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w", encoding="utf-8")

    try:
        with tmp_file as fp:
            # Recursively find all JSON files in the directory
            json_files = sorted(manifest_dir.rglob("*.json"))
            if not json_files:
                raise FileNotFoundError(f"No JSON files found in {manifest_dir}")

            print(f"[Tokenizer] Found {len(json_files)} JSON files to process...")

            for json_file in json_files:
                try:
                    with open(json_file, "r", encoding="utf-8") as handle:
                        data = json.load(handle)

                        # Extract text from WhisperX format
                        if "whisperx" in data and "utterances" in data["whisperx"]:
                            utterances = data["whisperx"]["utterances"]
                            for utterance in utterances:
                                text = utterance.get("text", "")
                                if not text.strip():
                                    num_empty += 1
                                    continue

                                # Normalize the text
                                normalized = normalizer.normalize(text, language=language)
                                if not normalized:
                                    num_empty += 1
                                    continue

                                fp.write(normalized + "\n")
                                num_samples += 1
                        else:
                            print(f"[Warning] Skipping {json_file}: unexpected format")

                except json.JSONDecodeError as e:
                    print(f"[Warning] Failed to parse {json_file}: {e}")
                    continue
                except Exception as e:
                    print(f"[Warning] Error processing {json_file}: {e}")
                    continue

    except Exception:
        os.unlink(tmp_file.name)
        raise

    return num_samples, num_empty, Path(tmp_file.name)


def train_tokenizer(args: argparse.Namespace) -> None:
    output_prefix = args.output_prefix.expanduser().resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    # Process input based on format
    if args.manifest_format == "whisperx":
        if not args.manifest_dir:
            raise ValueError("--manifest-dir is required for whisperx format")
        manifest_dir = args.manifest_dir.expanduser().resolve()
        if not manifest_dir.exists() or not manifest_dir.is_dir():
            raise FileNotFoundError(f"Manifest directory not found or not a directory: {manifest_dir}")

        print(f"[Tokenizer] Processing WhisperX JSON files from: {manifest_dir}")
        num_samples, num_empty, corpus_path = iter_texts_whisperx(manifest_dir, args.language)
    else:  # jsonl format
        if not args.manifest:
            raise ValueError("--manifest is required for jsonl format")
        manifests = [Path(m).expanduser().resolve() for m in args.manifest]
        missing = [str(p) for p in manifests if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Missing manifest(s): {', '.join(missing)}")

        print(f"[Tokenizer] Processing JSONL manifest files: {', '.join(str(m) for m in manifests)}")
        num_samples, num_empty, corpus_path = iter_texts_jsonl(manifests, args.language)

    if num_samples == 0:
        raise RuntimeError("No non-empty samples found. Cannot train tokenizer.")

    spm_kwargs = {
        "input": str(corpus_path),
        "model_prefix": str(output_prefix),
        "vocab_size": args.vocab_size,
        "character_coverage": args.character_coverage,
        "model_type": args.model_type,
        "bos_id": 0,
        "eos_id": 1,
        "unk_id": 2,
        "pad_id": -1,
        "input_sentence_size": args.input_sentence_size,
        "shuffle_input_sentence": True,
        "byte_fallback": args.byte_fallback,
        "train_extremely_large_corpus": True,
    }

    print(f"[Tokenizer] Training on {num_samples} samples (skipped {num_empty} empty).")
    print(f"[Tokenizer] Language: {args.language}, Vocab size: {args.vocab_size}, Coverage: {args.character_coverage}")
    try:
        spm.SentencePieceTrainer.train(**spm_kwargs)
    finally:
        corpus_path.unlink(missing_ok=True)

    model_path = output_prefix.with_suffix(".model")
    vocab_path = output_prefix.with_suffix(".vocab")

    print(f"[Tokenizer] Saved SentencePiece model to: {model_path}")
    print(f"[Tokenizer] Saved vocabulary to: {vocab_path}")


def main() -> int:
    args = parse_args()
    try:
        train_tokenizer(args)
    except KeyboardInterrupt:
        print("Interrupted!", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
