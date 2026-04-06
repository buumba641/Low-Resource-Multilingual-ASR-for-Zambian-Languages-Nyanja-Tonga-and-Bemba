"""
Text preprocessing utilities for Zambian language ASR.

Handles normalisation of Nyanja, Tonga and Bemba transcriptions
before vocabulary construction and model training.
"""

import re
import unicodedata
from typing import List


# Characters that are used as word-boundary tokens in CTC models
WORD_BOUNDARY_TOKEN = "|"


def unicode_normalize(text: str) -> str:
    """
    Apply NFC Unicode normalization.

    Args:
        text: Raw input text.

    Returns:
        NFC-normalized text.
    """
    return unicodedata.normalize("NFC", text)


def lowercase(text: str) -> str:
    """Convert text to lowercase."""
    return text.lower()


def remove_punctuation(text: str, keep_apostrophe: bool = True) -> str:
    """
    Remove punctuation characters from text.

    Zambian languages sometimes use apostrophes as part of word spelling,
    so they are retained by default.

    Args:
        text: Input text.
        keep_apostrophe: If True, retain apostrophe characters.

    Returns:
        Text with punctuation removed.
    """
    if keep_apostrophe:
        return re.sub(r"[^\w\s']", "", text)
    return re.sub(r"[^\w\s]", "", text)


def collapse_whitespace(text: str) -> str:
    """Replace consecutive whitespace characters with a single space."""
    return re.sub(r"\s+", " ", text).strip()


def replace_spaces_with_boundary(text: str) -> str:
    """
    Replace space characters with the CTC word-boundary token '|'.

    This is the canonical representation used by Wav2Vec2 CTC tokenisers.

    Args:
        text: Preprocessed transcription with space-separated words.

    Returns:
        Text with spaces replaced by '|'.
    """
    return text.replace(" ", WORD_BOUNDARY_TOKEN)


def normalize_transcription(
    text: str,
    for_training: bool = False,
) -> str:
    """
    Full normalisation pipeline for a single transcription.

    Steps:
      1. Unicode NFC normalization
      2. Lowercase conversion
      3. Punctuation removal (apostrophes kept)
      4. Whitespace collapsing
      5. Optionally replace spaces with '|' for CTC training targets

    Args:
        text: Raw transcription text.
        for_training: If True, spaces are replaced with '|'.

    Returns:
        Normalized transcription string.
    """
    text = unicode_normalize(text)
    text = lowercase(text)
    text = remove_punctuation(text, keep_apostrophe=True)
    text = collapse_whitespace(text)
    if for_training:
        text = replace_spaces_with_boundary(text)
    return text


def normalize_batch(
    texts: List[str],
    for_training: bool = False,
) -> List[str]:
    """
    Apply normalisation to a list of transcriptions.

    Args:
        texts: List of raw transcription strings.
        for_training: Passed through to normalize_transcription.

    Returns:
        List of normalized transcription strings.
    """
    return [normalize_transcription(t, for_training=for_training) for t in texts]


def preprocess_function(batch: dict, for_training: bool = False) -> dict:
    """
    Hugging Face Dataset map-compatible preprocessing function.

    Normalises the 'transcription' column in place.

    Args:
        batch: A batch dict from a Hugging Face Dataset.
        for_training: Whether to replace spaces with '|'.

    Returns:
        Batch dict with normalised 'transcription' values.
    """
    batch["transcription"] = [
        normalize_transcription(t, for_training=for_training)
        for t in batch["transcription"]
    ]
    return batch
