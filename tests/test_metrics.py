"""
Unit tests for src/utils/metrics.py
"""

import pytest

from src.utils.metrics import (
    compute_all_metrics,
    compute_cer,
    compute_wer,
    normalize_text,
    summarize_metrics,
)


class TestNormalizeText:
    def test_lowercase(self):
        assert normalize_text("Hello World") == "hello world"

    def test_strips_punctuation(self):
        assert normalize_text("hello, world!") == "hello world"

    def test_preserves_apostrophes(self):
        result = normalize_text("it's fine")
        assert "'" in result

    def test_collapses_whitespace(self):
        assert normalize_text("hello   world") == "hello world"

    def test_strips_leading_trailing(self):
        assert normalize_text("  hello  ") == "hello"

    def test_empty_string(self):
        assert normalize_text("") == ""


class TestComputeWer:
    def test_perfect_match(self):
        wer = compute_wer(["hello world"], ["hello world"])
        assert wer == pytest.approx(0.0)

    def test_complete_mismatch(self):
        wer = compute_wer(["foo bar"], ["baz qux"])
        assert wer == pytest.approx(1.0)

    def test_partial_match(self):
        wer = compute_wer(["hello world"], ["hello there"])
        assert 0.0 < wer <= 1.0

    def test_batch(self):
        predictions = ["hello world", "nyanja language"]
        references = ["hello world", "nyanja language"]
        wer = compute_wer(predictions, references)
        assert wer == pytest.approx(0.0)


class TestComputeCer:
    def test_perfect_match(self):
        cer = compute_cer(["abc"], ["abc"])
        assert cer == pytest.approx(0.0)

    def test_single_char_error(self):
        cer = compute_cer(["abc"], ["abd"])
        assert cer > 0.0

    def test_batch(self):
        predictions = ["tonga", "bemba"]
        references = ["tonga", "bemba"]
        cer = compute_cer(predictions, references)
        assert cer == pytest.approx(0.0)


class TestComputeAllMetrics:
    def test_returns_wer_and_cer(self):
        metrics = compute_all_metrics(["hello world"], ["hello world"])
        assert "wer" in metrics
        assert "cer" in metrics

    def test_perfect_predictions(self):
        metrics = compute_all_metrics(["hello"], ["hello"])
        assert metrics["wer"] == pytest.approx(0.0)
        assert metrics["cer"] == pytest.approx(0.0)

    def test_imperfect_predictions(self):
        metrics = compute_all_metrics(["foo"], ["bar"])
        assert metrics["wer"] > 0.0
        assert metrics["cer"] > 0.0


class TestSummarizeMetrics:
    def test_summarize_flat(self):
        results = {
            "nyanja": {"wer": 0.2, "cer": 0.1},
            "tonga": {"wer": 0.3, "cer": 0.15},
            "bemba": {"wer": 0.25, "cer": 0.12},
        }
        summary = summarize_metrics(results)
        assert "nyanja_wer" in summary
        assert "tonga_cer" in summary
        assert "avg_wer" in summary
        assert "avg_cer" in summary

    def test_avg_wer_correct(self):
        results = {
            "nyanja": {"wer": 0.2, "cer": 0.1},
            "tonga": {"wer": 0.4, "cer": 0.2},
        }
        summary = summarize_metrics(results)
        assert summary["avg_wer"] == pytest.approx(0.3)
        assert summary["avg_cer"] == pytest.approx(0.15)

    def test_empty_results(self):
        summary = summarize_metrics({})
        assert summary == {}
