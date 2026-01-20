#!/usr/bin/env python3
"""
Verify that a trained tokenizer doesn't produce <unk> tokens on your dataset.

Usage:
    python tools/tokenizer/verify_tokenizer.py \
        --tokenizer checkpoints/polish_bpe.model \
        --test-dir /workspace/10_hours/transcripts \
        --language pl \
        --sample-size 1000

Or test with specific text:
    python tools/tokenizer/verify_tokenizer.py \
        --tokenizer checkpoints/polish_bpe.model \
        --language pl \
        --text "wątpię nie było żadnych znaków"
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

from indextts.utils.front import TextNormalizer, TextTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify tokenizer quality and check for <unk> tokens")
    parser.add_argument(
        "--tokenizer",
        type=Path,
        required=True,
        help="Path to the .model file",
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        help="Directory with WhisperX JSON files to test",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="pl",
        help="Language code (e.g., 'pl', 'ja', 'zh', 'en')",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=1000,
        help="Number of samples to test (0 = all)",
    )
    parser.add_argument(
        "--text",
        type=str,
        help="Single text string to test (alternative to --test-dir)",
    )
    parser.add_argument(
        "--show-examples",
        action="store_true",
        help="Show examples of texts that produce <unk> tokens",
    )
    return parser.parse_args()


def load_tokenizer(model_path: Path, language: str) -> TextTokenizer:
    """Load the tokenizer with normalizer."""
    if not model_path.exists():
        raise FileNotFoundError(f"Tokenizer model not found: {model_path}")

    normalizer = TextNormalizer(preferred_language=language)
    tokenizer = TextTokenizer(vocab_file=str(model_path), normalizer=normalizer)

    print(f"Tokenizer loaded: {model_path}")
    print(f"  Vocabulary size: {tokenizer.vocab_size}")
    print(f"  UNK token: '{tokenizer.unk_token}' (ID: {tokenizer.unk_token_id})")
    print(f"  BOS token: '{tokenizer.bos_token}' (ID: {tokenizer.bos_token_id})")
    print(f"  EOS token: '{tokenizer.eos_token}' (ID: {tokenizer.eos_token_id})")
    print()

    return tokenizer


def test_single_text(tokenizer: TextTokenizer, text: str, language: str) -> Tuple[bool, List[str], List[int]]:
    """Test a single text and return whether it contains <unk>."""
    tokens = tokenizer.tokenize(text, language=language)
    token_ids = tokenizer.encode(text, language=language)

    has_unk = tokenizer.unk_token_id in token_ids

    return has_unk, tokens, token_ids


def collect_texts_from_whisperx(test_dir: Path, sample_size: int = 0) -> List[str]:
    """Collect texts from WhisperX JSON files."""
    texts = []
    json_files = sorted(test_dir.rglob("*.json"))

    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {test_dir}")

    print(f"Found {len(json_files)} JSON files...")

    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            if "whisperx" in data and "utterances" in data["whisperx"]:
                for utterance in data["whisperx"]["utterances"]:
                    text = utterance.get("text", "").strip()
                    if text:
                        texts.append(text)
                        if sample_size > 0 and len(texts) >= sample_size:
                            return texts
        except Exception as e:
            print(f"[Warning] Error reading {json_file}: {e}")
            continue

    return texts


def main() -> int:
    args = parse_args()

    # Load tokenizer
    tokenizer = load_tokenizer(args.tokenizer, args.language)

    # Test single text if provided
    if args.text:
        print(f"Testing single text: '{args.text}'")
        print("-" * 80)

        has_unk, tokens, token_ids = test_single_text(tokenizer, args.text, args.language)

        print(f"Original text: {args.text}")
        print(f"Normalized: {tokenizer.normalizer.normalize(args.text, language=args.language)}")
        print(f"Tokens ({len(tokens)}): {tokens}")
        print(f"Token IDs: {token_ids}")
        print(f"Contains <unk>: {'YES ⚠️' if has_unk else 'NO ✓'}")

        # Decode to verify round-trip
        decoded = tokenizer.decode(token_ids)
        print(f"Decoded: {decoded}")
        print()

        return 0 if not has_unk else 1

    # Test directory of files
    if not args.test_dir:
        print("Error: Either --text or --test-dir must be specified")
        return 1

    print(f"Collecting texts from: {args.test_dir}")
    texts = collect_texts_from_whisperx(args.test_dir, args.sample_size)

    if not texts:
        print("No texts found to test!")
        return 1

    print(f"Testing {len(texts)} samples...\n")

    # Test all texts
    total = len(texts)
    texts_with_unk = []
    unk_examples = []

    for i, text in enumerate(texts):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{total} samples...")

        has_unk, tokens, token_ids = test_single_text(tokenizer, text, args.language)

        if has_unk:
            texts_with_unk.append(text)
            if args.show_examples and len(unk_examples) < 10:
                unk_examples.append({
                    "text": text,
                    "tokens": tokens,
                    "token_ids": token_ids
                })

    # Results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Total texts tested: {total}")
    print(f"Texts with <unk>: {len(texts_with_unk)} ({len(texts_with_unk)/total*100:.2f}%)")
    print(f"Texts without <unk>: {total - len(texts_with_unk)} ({(total-len(texts_with_unk))/total*100:.2f}%)")

    if len(texts_with_unk) == 0:
        print("\n✓ SUCCESS: No <unk> tokens found!")
        print("  Your tokenizer handles all characters in the dataset.")
        return 0
    else:
        print(f"\n⚠️  WARNING: {len(texts_with_unk)} texts contain <unk> tokens")
        print("  Consider:")
        print("    1. Increasing --character-coverage (e.g., 0.9999)")
        print("    2. Using --byte-fallback during training")
        print("    3. Increasing --vocab-size")

        if args.show_examples and unk_examples:
            print("\n" + "-" * 80)
            print("EXAMPLES OF TEXTS WITH <UNK>:")
            print("-" * 80)
            for i, example in enumerate(unk_examples, 1):
                print(f"\nExample {i}:")
                print(f"  Text: {example['text'][:100]}...")
                print(f"  Tokens: {example['tokens']}")
                unk_positions = [j for j, tid in enumerate(example['token_ids'])
                                if tid == tokenizer.unk_token_id]
                print(f"  <unk> at positions: {unk_positions}")

        return 1


if __name__ == "__main__":
    sys.exit(main())
