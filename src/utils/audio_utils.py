"""
Audio processing utilities for ASR data preparation.
"""

import os
from pathlib import Path
from typing import Optional, Tuple

import librosa
import numpy as np
import soundfile as sf


SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def load_audio(
    file_path: str,
    target_sample_rate: int = 16000,
) -> Tuple[np.ndarray, int]:
    """
    Load an audio file and resample to the target sample rate.

    Args:
        file_path: Path to the audio file.
        target_sample_rate: Target sample rate in Hz.

    Returns:
        Tuple of (audio array, sample rate).

    Raises:
        FileNotFoundError: If the audio file does not exist.
        ValueError: If the file format is not supported.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported audio format '{path.suffix}'. "
            f"Supported formats: {SUPPORTED_EXTENSIONS}"
        )
    audio, sample_rate = librosa.load(file_path, sr=target_sample_rate, mono=True)
    return audio, sample_rate


def get_audio_duration(file_path: str) -> float:
    """
    Get the duration of an audio file in seconds.

    Args:
        file_path: Path to the audio file.

    Returns:
        Duration in seconds.
    """
    audio, sr = load_audio(file_path)
    return len(audio) / sr


def filter_audio_by_duration(
    file_path: str,
    min_duration: float = 0.5,
    max_duration: float = 20.0,
) -> bool:
    """
    Check whether an audio file's duration falls within acceptable bounds.

    Args:
        file_path: Path to the audio file.
        min_duration: Minimum acceptable duration in seconds.
        max_duration: Maximum acceptable duration in seconds.

    Returns:
        True if duration is within bounds, False otherwise.
    """
    try:
        duration = get_audio_duration(file_path)
        return min_duration <= duration <= max_duration
    except Exception:
        return False


def save_audio(
    audio: np.ndarray,
    file_path: str,
    sample_rate: int = 16000,
) -> None:
    """
    Save a numpy audio array to a WAV file.

    Args:
        audio: Audio signal as a numpy array.
        file_path: Destination file path.
        sample_rate: Sample rate in Hz.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    sf.write(file_path, audio, sample_rate)


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """
    Normalize audio to the range [-1, 1].

    Args:
        audio: Input audio array.

    Returns:
        Normalized audio array.
    """
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        return audio / max_val
    return audio


def compute_audio_stats(audio: np.ndarray, sample_rate: int) -> dict:
    """
    Compute basic statistics for an audio signal.

    Args:
        audio: Audio signal as a numpy array.
        sample_rate: Sample rate in Hz.

    Returns:
        Dictionary with duration, rms, and peak amplitude.
    """
    duration = len(audio) / sample_rate
    rms = float(np.sqrt(np.mean(audio ** 2)))
    peak = float(np.max(np.abs(audio)))
    return {"duration_seconds": duration, "rms": rms, "peak": peak}
