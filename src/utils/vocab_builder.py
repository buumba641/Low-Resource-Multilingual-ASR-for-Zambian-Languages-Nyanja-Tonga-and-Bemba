"""
Vocabulary building utilities for CTC-based ASR models.
Extracts character vocabularies from transcription text.
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Set


SPECIAL_TOKENS = {
    "[PAD]": 0,
    "[UNK]": 1,
    "|": 2,
}


def extract_characters(transcriptions: List[str]) -> Set[str]:
    """
    Extract unique characters from a list of transcriptions.

    Args:
        transcriptions: List of transcription strings.

    Returns:
        Set of unique characters found in the transcriptions.
    """
    chars: Set[str] = set()
    for text in transcriptions:
        text = text.lower().strip()
        text = re.sub(r"\s+", " ", text)
        chars.update(set(text))
    chars.discard(" ")
    return chars


def build_vocab(
    transcriptions: List[str],
    extra_transcriptions: Optional[List[List[str]]] = None,
) -> Dict[str, int]:
    """
    Build a character-level vocabulary from transcriptions.

    The vocabulary always starts with special tokens: [PAD]=0, [UNK]=1, |=2.
    The word boundary token '|' represents spaces between words.

    Args:
        transcriptions: Primary list of transcription strings.
        extra_transcriptions: Optional additional lists to include in vocab.

    Returns:
        Dictionary mapping characters to integer token IDs.
    """
    all_transcriptions = list(transcriptions)
    if extra_transcriptions:
        for extra in extra_transcriptions:
            all_transcriptions.extend(extra)

    chars = extract_characters(all_transcriptions)
    vocab = dict(SPECIAL_TOKENS)
    next_id = max(vocab.values()) + 1
    for char in sorted(chars):
        if char not in vocab:
            vocab[char] = next_id
            next_id += 1
    return vocab


def save_vocab(vocab: Dict[str, int], output_path: str) -> None:
    """
    Save a vocabulary dictionary to a JSON file.

    Args:
        vocab: Dictionary mapping tokens to integer IDs.
        output_path: Destination file path for the JSON file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)


def load_vocab(vocab_path: str) -> Dict[str, int]:
    """
    Load a vocabulary dictionary from a JSON file.

    Args:
        vocab_path: Path to the vocabulary JSON file.

    Returns:
        Dictionary mapping tokens to integer IDs.

    Raises:
        FileNotFoundError: If the vocab file does not exist.
    """
    path = Path(vocab_path)
    if not path.exists():
        raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")
    with open(vocab_path, "r", encoding="utf-8") as f:
        return json.load(f)


def merge_vocabs(vocabs: List[Dict[str, int]]) -> Dict[str, int]:
    """
    Merge multiple vocabulary dictionaries into a single unified vocabulary.
    Special tokens are preserved at their original positions.

    Args:
        vocabs: List of vocabulary dictionaries to merge.

    Returns:
        Merged vocabulary dictionary with consistent token IDs.
    """
    all_chars: Set[str] = set()
    for vocab in vocabs:
        for token in vocab:
            if token not in SPECIAL_TOKENS:
                all_chars.add(token)

    merged = dict(SPECIAL_TOKENS)
    next_id = max(merged.values()) + 1
    for char in sorted(all_chars):
        if char not in merged:
            merged[char] = next_id
            next_id += 1
    return merged
