"""
Unit tests for src/data_preparation/prepare_dataset.py
"""

import csv
import json
import os
import tempfile

import numpy as np
import pytest
import soundfile as sf

from src.data_preparation.prepare_dataset import (
    load_manifest,
    scan_audio_directory,
    split_samples,
)


def make_wav_file(path: str, duration: float = 2.0, sr: int = 16000) -> None:
    """Create a synthetic WAV file at the given path."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    samples = np.zeros(int(sr * duration), dtype=np.float32)
    sf.write(path, samples, sr)


class TestLoadManifest:
    def test_valid_csv(self, tmp_path):
        csv_path = tmp_path / "manifest.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["audio_path", "transcription"])
            writer.writeheader()
            writer.writerow({"audio_path": "a.wav", "transcription": "hello"})
            writer.writerow({"audio_path": "b.wav", "transcription": "world"})
        samples = load_manifest(str(csv_path))
        assert len(samples) == 2
        assert samples[0]["audio_path"] == "a.wav"
        assert samples[1]["transcription"] == "world"

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_manifest("/nonexistent/manifest.csv")

    def test_missing_columns_raises(self, tmp_path):
        csv_path = tmp_path / "bad.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["audio_path"])
            writer.writeheader()
            writer.writerow({"audio_path": "a.wav"})
        with pytest.raises(ValueError, match="transcription"):
            load_manifest(str(csv_path))

    def test_tsv_format(self, tmp_path):
        tsv_path = tmp_path / "manifest.tsv"
        with open(tsv_path, "w", newline="", encoding="utf-8") as f:
            f.write("audio_path\ttranscription\n")
            f.write("a.wav\thello nyanja\n")
        samples = load_manifest(str(tsv_path))
        assert len(samples) == 1
        assert samples[0]["transcription"] == "hello nyanja"


class TestScanAudioDirectory:
    def test_basic_scan(self, tmp_path):
        audio_dir = tmp_path / "audio"
        trans_dir = tmp_path / "transcriptions"
        audio_dir.mkdir()
        trans_dir.mkdir()

        make_wav_file(str(audio_dir / "sample1.wav"), duration=2.0)
        (trans_dir / "sample1.txt").write_text("ndiyo", encoding="utf-8")

        samples = scan_audio_directory(
            str(audio_dir), str(trans_dir), language="nyanja"
        )
        assert len(samples) == 1
        assert samples[0]["transcription"] == "ndiyo"
        assert samples[0]["language"] == "nyanja"

    def test_missing_transcription_skipped(self, tmp_path):
        audio_dir = tmp_path / "audio"
        trans_dir = tmp_path / "transcriptions"
        audio_dir.mkdir()
        trans_dir.mkdir()

        make_wav_file(str(audio_dir / "orphan.wav"), duration=2.0)

        samples = scan_audio_directory(str(audio_dir), str(trans_dir), language="tonga")
        assert len(samples) == 0

    def test_too_short_skipped(self, tmp_path):
        audio_dir = tmp_path / "audio"
        trans_dir = tmp_path / "transcriptions"
        audio_dir.mkdir()
        trans_dir.mkdir()

        make_wav_file(str(audio_dir / "tiny.wav"), duration=0.1)
        (trans_dir / "tiny.txt").write_text("short", encoding="utf-8")

        samples = scan_audio_directory(
            str(audio_dir), str(trans_dir), language="bemba", min_duration=0.5
        )
        assert len(samples) == 0

    def test_empty_transcription_skipped(self, tmp_path):
        audio_dir = tmp_path / "audio"
        trans_dir = tmp_path / "transcriptions"
        audio_dir.mkdir()
        trans_dir.mkdir()

        make_wav_file(str(audio_dir / "empty.wav"), duration=2.0)
        (trans_dir / "empty.txt").write_text("", encoding="utf-8")

        samples = scan_audio_directory(str(audio_dir), str(trans_dir), language="nyanja")
        assert len(samples) == 0


class TestSplitSamples:
    def _make_samples(self, n: int):
        return [{"audio_path": f"a{i}.wav", "transcription": f"text{i}"} for i in range(n)]

    def test_split_proportions(self):
        samples = self._make_samples(100)
        train, val, test = split_samples(samples, train_ratio=0.8, val_ratio=0.1)
        assert len(train) == 80
        assert len(val) == 10
        assert len(test) == 10

    def test_total_count_preserved(self):
        samples = self._make_samples(50)
        train, val, test = split_samples(samples)
        assert len(train) + len(val) + len(test) == 50

    def test_reproducible_with_seed(self):
        samples = self._make_samples(30)
        t1, v1, te1 = split_samples(samples, seed=42)
        t2, v2, te2 = split_samples(samples, seed=42)
        assert [s["audio_path"] for s in t1] == [s["audio_path"] for s in t2]

    def test_different_seeds_differ(self):
        samples = self._make_samples(30)
        t1, _, _ = split_samples(samples, seed=1)
        t2, _, _ = split_samples(samples, seed=99)
        assert [s["audio_path"] for s in t1] != [s["audio_path"] for s in t2]

    def test_small_dataset(self):
        samples = self._make_samples(3)
        train, val, test = split_samples(samples, train_ratio=0.8, val_ratio=0.1)
        assert len(train) + len(val) + len(test) == 3
