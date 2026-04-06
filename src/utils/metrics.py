"""
Metrics utilities for ASR evaluation.
Computes Word Error Rate (WER) and Character Error Rate (CER) using jiwer.
"""

import re
from typing import List, Dict, Optional
import numpy as np
import jiwer


# Standard jiwer transformation pipeline for WER
_WER_TRANSFORM = jiwer.Compose(
    [
        jiwer.ToLowerCase(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.ReduceToListOfListOfWords(),
    ]
)

# Character-level transformation for CER
_CER_TRANSFORM = jiwer.Compose(
    [
        jiwer.ToLowerCase(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.ReduceToListOfListOfChars(),
    ]
)


def normalize_text(text: str) -> str:
    """
    Normalize transcription text for evaluation.
    Converts to lowercase and removes punctuation except apostrophes.
    """
    text = text.lower().strip()
    text = re.sub(r"[^\w\s']", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_wer(predictions: List[str], references: List[str], normalize: bool = True) -> float:
    """
    Compute Word Error Rate between predictions and references.

    Args:
        predictions: List of predicted transcriptions.
        references: List of ground-truth transcriptions.
        normalize: Whether to normalize text before computing WER.

    Returns:
        WER score as a float (0.0 = perfect, 1.0+ = poor).
    """
    if normalize:
        predictions = [normalize_text(p) for p in predictions]
        references = [normalize_text(r) for r in references]
    return jiwer.wer(
        references,
        predictions,
        reference_transform=_WER_TRANSFORM,
        hypothesis_transform=_WER_TRANSFORM,
    )


def compute_cer(predictions: List[str], references: List[str], normalize: bool = True) -> float:
    """
    Compute Character Error Rate between predictions and references.

    Args:
        predictions: List of predicted transcriptions.
        references: List of ground-truth transcriptions.
        normalize: Whether to normalize text before computing CER.

    Returns:
        CER score as a float (0.0 = perfect, 1.0+ = poor).
    """
    if normalize:
        predictions = [normalize_text(p) for p in predictions]
        references = [normalize_text(r) for r in references]
    return jiwer.cer(
        references,
        predictions,
        reference_transform=_CER_TRANSFORM,
        hypothesis_transform=_CER_TRANSFORM,
    )


def compute_all_metrics(
    predictions: List[str],
    references: List[str],
    normalize: bool = True,
) -> Dict[str, float]:
    """
    Compute WER and CER together.

    Args:
        predictions: List of predicted transcriptions.
        references: List of ground-truth transcriptions.
        normalize: Whether to normalize text before computing metrics.

    Returns:
        Dictionary with 'wer' and 'cer' keys.
    """
    if normalize:
        predictions = [normalize_text(p) for p in predictions]
        references = [normalize_text(r) for r in references]
    wer = jiwer.wer(
        references,
        predictions,
        reference_transform=_WER_TRANSFORM,
        hypothesis_transform=_WER_TRANSFORM,
    )
    cer = jiwer.cer(
        references,
        predictions,
        reference_transform=_CER_TRANSFORM,
        hypothesis_transform=_CER_TRANSFORM,
    )
    return {"wer": wer, "cer": cer}


def compute_batch_metrics(
    predictions: List[str],
    references: List[str],
    language: Optional[str] = None,
    normalize: bool = True,
) -> Dict[str, float]:
    """
    Compute metrics for a batch and return a summary dict.

    Args:
        predictions: List of predicted transcriptions.
        references: List of ground-truth transcriptions.
        language: Optional language label for logging.
        normalize: Whether to normalize text.

    Returns:
        Dictionary with metric names as keys and float scores as values.
    """
    metrics = compute_all_metrics(predictions, references, normalize=normalize)
    if language:
        metrics = {f"{language}_{k}": v for k, v in metrics.items()}
    return metrics


def summarize_metrics(results: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """
    Summarize per-language metrics into a flat dictionary.

    Args:
        results: Nested dict mapping language -> {metric -> value}.

    Returns:
        Flat dict of all metrics plus macro-average WER and CER.
    """
    flat: Dict[str, float] = {}
    wer_values = []
    cer_values = []
    for lang, metrics in results.items():
        for metric_name, value in metrics.items():
            flat[f"{lang}_{metric_name}"] = value
        if "wer" in metrics:
            wer_values.append(metrics["wer"])
        if "cer" in metrics:
            cer_values.append(metrics["cer"])
    if wer_values:
        flat["avg_wer"] = float(np.mean(wer_values))
    if cer_values:
        flat["avg_cer"] = float(np.mean(cer_values))
    return flat
