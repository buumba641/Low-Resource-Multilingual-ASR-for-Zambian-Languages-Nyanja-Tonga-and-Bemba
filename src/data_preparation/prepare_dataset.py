"""
Dataset preparation for monolingual and multilingual ASR.

Expects data in the following directory layout:
    data/<language>/
        audio/          - audio files (.wav, .mp3, .flac)
        transcriptions/ - matching .txt transcription files (same stem)

OR a single manifest CSV/TSV with columns:
    audio_path, transcription [, language]

The script splits data into train/validation/test sets, builds a character
vocabulary, and saves Hugging Face Dataset objects to disk.
"""

import csv
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml
from datasets import Audio, Dataset, DatasetDict

from src.utils.audio_utils import filter_audio_by_duration
from src.utils.vocab_builder import build_vocab, save_vocab

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_manifest(manifest_path: str) -> List[Dict]:
    """
    Load a CSV/TSV manifest file into a list of sample dicts.

    The manifest must have at minimum the columns 'audio_path' and
    'transcription'. An optional 'language' column is also supported.

    Args:
        manifest_path: Path to the manifest CSV or TSV file.

    Returns:
        List of dicts with at least 'audio_path' and 'transcription' keys.

    Raises:
        FileNotFoundError: If the manifest file does not exist.
        ValueError: If required columns are missing.
    """
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    samples = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        required = {"audio_path", "transcription"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(
                f"Manifest must contain columns: {required}. "
                f"Found: {reader.fieldnames}"
            )
        for row in reader:
            samples.append(dict(row))
    return samples


def scan_audio_directory(
    audio_dir: str,
    transcription_dir: str,
    language: str,
    min_duration: float = 0.5,
    max_duration: float = 20.0,
) -> List[Dict]:
    """
    Scan an audio directory for supported audio files and pair each with its
    corresponding transcription file.

    Args:
        audio_dir: Directory containing audio files.
        transcription_dir: Directory containing .txt transcription files.
        language: Language label to attach to each sample.
        min_duration: Minimum audio duration to keep (seconds).
        max_duration: Maximum audio duration to keep (seconds).

    Returns:
        List of sample dicts with 'audio_path', 'transcription', and 'language'.
    """
    audio_dir = Path(audio_dir)
    transcription_dir = Path(transcription_dir)
    supported = {".wav", ".mp3", ".flac", ".ogg"}
    samples = []
    skipped = 0

    for audio_file in sorted(audio_dir.iterdir()):
        if audio_file.suffix.lower() not in supported:
            continue
        txt_file = transcription_dir / (audio_file.stem + ".txt")
        if not txt_file.exists():
            logger.warning("No transcription for %s — skipping.", audio_file.name)
            skipped += 1
            continue
        if not filter_audio_by_duration(str(audio_file), min_duration, max_duration):
            logger.warning("Duration out of bounds for %s — skipping.", audio_file.name)
            skipped += 1
            continue
        transcription = txt_file.read_text(encoding="utf-8").strip()
        if not transcription:
            logger.warning("Empty transcription for %s — skipping.", audio_file.name)
            skipped += 1
            continue
        samples.append(
            {
                "audio_path": str(audio_file),
                "transcription": transcription,
                "language": language,
            }
        )

    logger.info(
        "Loaded %d samples for '%s' (%d skipped).", len(samples), language, skipped
    )
    return samples


def split_samples(
    samples: List[Dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Randomly split samples into train, validation, and test sets.

    Args:
        samples: Full list of samples.
        train_ratio: Fraction of data for training.
        val_ratio: Fraction of data for validation (remainder goes to test).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train, validation, test) sample lists.
    """
    import random

    rng = random.Random(seed)
    shuffled = list(samples)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = shuffled[:n_train]
    val = shuffled[n_train : n_train + n_val]
    test = shuffled[n_train + n_val :]
    return train, val, test


def samples_to_dataset(samples: List[Dict], sampling_rate: int = 16000) -> Dataset:
    """
    Convert a list of sample dicts to a Hugging Face Dataset with audio loading.

    Args:
        samples: List of sample dicts (must contain 'audio_path' and 'transcription').
        sampling_rate: Target audio sampling rate.

    Returns:
        A Hugging Face Dataset with an 'audio' column decoded at the given rate.
    """
    records = {
        "audio": [s["audio_path"] for s in samples],
        "transcription": [s["transcription"] for s in samples],
    }
    if samples and "language" in samples[0]:
        records["language"] = [s["language"] for s in samples]

    dataset = Dataset.from_dict(records)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=sampling_rate))
    return dataset


def prepare_monolingual_dataset(
    config_path: str,
    output_dir: Optional[str] = None,
) -> DatasetDict:
    """
    Prepare a monolingual dataset for a single language based on a YAML config.

    Expects the config to point to a directory with 'audio/' and
    'transcriptions/' subdirectories, or a 'manifest.csv' file.

    Args:
        config_path: Path to the language YAML config file.
        output_dir: Directory to save the processed DatasetDict. If None,
                    uses the output_dir from the config.

    Returns:
        DatasetDict with 'train', 'validation', and 'test' splits.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    language = config["language"]
    dataset_path = Path(config["dataset_path"])
    output_dir = output_dir or config.get("output_dir", f"outputs/{language}")
    sampling_rate = config["audio"]["sampling_rate"]
    min_dur = config["audio"]["min_duration_seconds"]
    max_dur = config["audio"]["max_duration_seconds"]
    train_ratio = config["data_split"]["train"]
    val_ratio = config["data_split"]["validation"]
    seed = config["training"].get("seed", 42)

    manifest_path = dataset_path / "manifest.csv"
    if manifest_path.exists():
        samples = load_manifest(str(manifest_path))
        for s in samples:
            s.setdefault("language", language)
    else:
        audio_dir = dataset_path / "audio"
        transcription_dir = dataset_path / "transcriptions"
        if not audio_dir.exists() or not transcription_dir.exists():
            raise FileNotFoundError(
                f"Expected 'audio/' and 'transcriptions/' in {dataset_path}, "
                "or a 'manifest.csv' file."
            )
        samples = scan_audio_directory(
            str(audio_dir),
            str(transcription_dir),
            language=language,
            min_duration=min_dur,
            max_duration=max_dur,
        )

    if not samples:
        raise ValueError(f"No valid samples found for language '{language}'.")

    train_samples, val_samples, test_samples = split_samples(
        samples, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed
    )
    logger.info(
        "%s split — train: %d, val: %d, test: %d",
        language,
        len(train_samples),
        len(val_samples),
        len(test_samples),
    )

    vocab_output = os.path.join(output_dir, "vocab.json")
    all_texts = [s["transcription"] for s in samples]
    vocab = build_vocab(all_texts)
    os.makedirs(output_dir, exist_ok=True)
    save_vocab(vocab, vocab_output)
    logger.info("Saved vocabulary (%d tokens) to %s.", len(vocab), vocab_output)

    dataset_dict = DatasetDict(
        {
            "train": samples_to_dataset(train_samples, sampling_rate),
            "validation": samples_to_dataset(val_samples, sampling_rate),
            "test": samples_to_dataset(test_samples, sampling_rate),
        }
    )

    processed_path = os.path.join(output_dir, "dataset")
    dataset_dict.save_to_disk(processed_path)
    logger.info("Saved dataset to %s.", processed_path)
    return dataset_dict


def prepare_multilingual_dataset(
    config_path: str,
    output_dir: Optional[str] = None,
) -> DatasetDict:
    """
    Prepare a combined multilingual dataset from multiple language configs.

    All languages are pooled together. A shared vocabulary is built from
    all transcriptions. Language labels are preserved in the 'language' column.

    Args:
        config_path: Path to the multilingual YAML config file.
        output_dir: Directory to save the processed DatasetDict. If None,
                    uses the output_dir from the config.

    Returns:
        DatasetDict with 'train', 'validation', and 'test' splits.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    output_dir = output_dir or config.get("output_dir", "outputs/multilingual")
    sampling_rate = config["audio"]["sampling_rate"]
    min_dur = config["audio"]["min_duration_seconds"]
    max_dur = config["audio"]["max_duration_seconds"]
    train_ratio = config["data_split"]["train"]
    val_ratio = config["data_split"]["validation"]
    seed = config["training"].get("seed", 42)

    all_train, all_val, all_test = [], [], []
    per_lang_texts: List[List[str]] = []

    for lang_cfg in config["languages"]:
        language = lang_cfg["name"]
        dataset_path = Path(lang_cfg["dataset_path"])
        manifest_path = dataset_path / "manifest.csv"

        if manifest_path.exists():
            samples = load_manifest(str(manifest_path))
            for s in samples:
                s.setdefault("language", language)
        else:
            audio_dir = dataset_path / "audio"
            transcription_dir = dataset_path / "transcriptions"
            if not audio_dir.exists() or not transcription_dir.exists():
                logger.warning(
                    "Skipping '%s': missing audio/ or transcriptions/ in %s.",
                    language,
                    dataset_path,
                )
                continue
            samples = scan_audio_directory(
                str(audio_dir),
                str(transcription_dir),
                language=language,
                min_duration=min_dur,
                max_duration=max_dur,
            )

        if not samples:
            logger.warning("No valid samples for '%s' — skipping.", language)
            continue

        train_s, val_s, test_s = split_samples(
            samples, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed
        )
        all_train.extend(train_s)
        all_val.extend(val_s)
        all_test.extend(test_s)
        per_lang_texts.append([s["transcription"] for s in samples])

    if not all_train:
        raise ValueError("No valid samples found across any language.")

    primary_texts = [s["transcription"] for s in all_train + all_val + all_test]
    vocab = build_vocab(primary_texts, extra_transcriptions=per_lang_texts)
    os.makedirs(output_dir, exist_ok=True)
    vocab_output = os.path.join(output_dir, "vocab.json")
    save_vocab(vocab, vocab_output)
    logger.info(
        "Saved multilingual vocabulary (%d tokens) to %s.", len(vocab), vocab_output
    )

    dataset_dict = DatasetDict(
        {
            "train": samples_to_dataset(all_train, sampling_rate),
            "validation": samples_to_dataset(all_val, sampling_rate),
            "test": samples_to_dataset(all_test, sampling_rate),
        }
    )

    processed_path = os.path.join(output_dir, "dataset")
    dataset_dict.save_to_disk(processed_path)
    logger.info(
        "Saved multilingual dataset (train=%d, val=%d, test=%d) to %s.",
        len(all_train),
        len(all_val),
        len(all_test),
        processed_path,
    )
    return dataset_dict


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare ASR dataset.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    parser.add_argument("--multilingual", action="store_true", help="Prepare multilingual dataset.")
    parser.add_argument("--output_dir", default=None, help="Override output directory.")
    args = parser.parse_args()

    if args.multilingual:
        prepare_multilingual_dataset(args.config, args.output_dir)
    else:
        prepare_monolingual_dataset(args.config, args.output_dir)
