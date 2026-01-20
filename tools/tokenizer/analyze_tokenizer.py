#!/usr/bin/env python3
"""
Analyze tokenizer statistics and character coverage.

Usage:
    python tools/tokenizer/analyze_tokenizer.py \
        --tokenizer checkpoints/polish_bpe.model \
        --language pl
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

from indextts.utils.front import TextNormalizer, TextTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze tokenizer statistics")
    parser.add_argument(
        "--tokenizer",
        type=Path,
        required=True,
        help="Path to the .model file",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="pl",
        help="Language code",
    )
    return parser.parse_args()


def analyze_vocabulary(tokenizer: TextTokenizer):
    """Analyze the tokenizer vocabulary."""
    vocab = tokenizer.get_vocab()

    print("=" * 80)
    print("VOCABULARY STATISTICS")
    print("=" * 80)
    print(f"Total vocabulary size: {len(vocab)}")
    print(f"Special tokens:")
    print(f"  <s> (BOS): {tokenizer.bos_token_id}")
    print(f"  </s> (EOS): {tokenizer.eos_token_id}")
    print(f"  <unk> (UNK): {tokenizer.unk_token_id}")
    print()

    # Analyze token types
    single_chars = []
    subwords = []
    words = []
    special = []

    for token, idx in vocab.items():
        if token in ["<s>", "</s>", "<unk>", "<pad>"]:
            special.append(token)
        elif len(token) == 1:
            single_chars.append(token)
        elif token.startswith("▁"):
            if len(token) > 2:
                words.append(token)
            else:
                single_chars.append(token)
        else:
            subwords.append(token)

    print(f"Token distribution:")
    print(f"  Special tokens: {len(special)}")
    print(f"  Single characters: {len(single_chars)}")
    print(f"  Subword pieces: {len(subwords)}")
    print(f"  Complete words: {len(words)}")
    print()

    # Show sample tokens
    print("Sample tokens:")
    all_tokens = sorted(vocab.items(), key=lambda x: x[1])[:50]
    for token, idx in all_tokens[:20]:
        print(f"  [{idx:5d}] {repr(token)}")
    print()


def test_polish_characters(tokenizer: TextTokenizer, language: str):
    """Test Polish-specific characters."""
    polish_chars = "aąbcćdeęfghijklłmnńoóprsśtuwyzźż"
    polish_upper = polish_chars.upper()

    print("=" * 80)
    print("POLISH CHARACTER COVERAGE")
    print("=" * 80)

    # Test lowercase
    print("Lowercase Polish characters:")
    for char in polish_chars:
        tokens = tokenizer.tokenize(char, language=language)
        ids = tokenizer.encode(char, language=language)
        has_unk = tokenizer.unk_token_id in ids
        status = "❌ UNK" if has_unk else "✓"
        print(f"  {status} '{char}' -> {tokens} (ID: {ids})")

    print()

    # Test uppercase
    print("Uppercase Polish characters:")
    for char in polish_upper:
        tokens = tokenizer.tokenize(char, language=language)
        ids = tokenizer.encode(char, language=language)
        has_unk = tokenizer.unk_token_id in ids
        status = "❌ UNK" if has_unk else "✓"
        print(f"  {status} '{char}' -> {tokens} (ID: {ids})")

    print()


def test_common_words(tokenizer: TextTokenizer, language: str):
    """Test common Polish words."""
    common_words = [
        "tak",
        "nie",
        "jest",
        "jestem",
        "dzień",
        "dobry",
        "dziękuję",
        "proszę",
        "przepraszam",
        "miłość",
        "życie",
        "świat",
        "Polska",
        "człowiek",
        "wątpię",
        "był",
        "było",
        "żaden",
    ]

    print("=" * 80)
    print("COMMON POLISH WORDS TOKENIZATION")
    print("=" * 80)

    for word in common_words:
        tokens = tokenizer.tokenize(word, language=language)
        ids = tokenizer.encode(word, language=language)
        has_unk = tokenizer.unk_token_id in ids
        status = "❌" if has_unk else "✓"
        print(f"{status} '{word:15s}' -> {len(tokens):2d} tokens: {tokens}")

    print()


def test_sentence_tokenization(tokenizer: TextTokenizer, language: str):
    """Test full sentence tokenization."""
    sentences = [
        "Dzień dobry, jak się masz?",
        "Polska jest pięknym krajem.",
        "Wątpię, czy nie było żadnych znaków.",
        "Dziękuję bardzo za pomoc!",
        "To jest test tokenizacji zdań.",
    ]

    print("=" * 80)
    print("SENTENCE TOKENIZATION")
    print("=" * 80)

    for sentence in sentences:
        tokens = tokenizer.tokenize(sentence, language=language)
        ids = tokenizer.encode(sentence, language=language)
        has_unk = tokenizer.unk_token_id in ids

        print(f"Original: {sentence}")
        print(f"  Tokens ({len(tokens)}): {tokens}")
        print(f"  IDs: {ids[:20]}{'...' if len(ids) > 20 else ''}")
        print(f"  Has <unk>: {'YES ❌' if has_unk else 'NO ✓'}")

        # Decode
        decoded = tokenizer.decode(ids)
        print(f"  Decoded: {decoded}")
        print()


def main() -> int:
    args = parse_args()

    # Load tokenizer
    if not args.tokenizer.exists():
        print(f"Error: Tokenizer not found: {args.tokenizer}")
        return 1

    normalizer = TextNormalizer(preferred_language=args.language)
    tokenizer = TextTokenizer(vocab_file=str(args.tokenizer), normalizer=normalizer)

    print(f"\nAnalyzing tokenizer: {args.tokenizer}")
    print(f"Language: {args.language}\n")

    # Run analyses
    analyze_vocabulary(tokenizer)
    test_polish_characters(tokenizer, args.language)
    test_common_words(tokenizer, args.language)
    test_sentence_tokenization(tokenizer, args.language)

    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
