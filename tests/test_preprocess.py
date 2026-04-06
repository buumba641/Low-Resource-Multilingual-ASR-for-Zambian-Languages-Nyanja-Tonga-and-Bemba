"""
Unit tests for src/data_preparation/preprocess.py
"""

import pytest

from src.data_preparation.preprocess import (
    collapse_whitespace,
    lowercase,
    normalize_batch,
    normalize_transcription,
    preprocess_function,
    remove_punctuation,
    replace_spaces_with_boundary,
    unicode_normalize,
)


class TestUnicodeNormalize:
    def test_nfc_normalization(self):
        # Composed form: é (U+00E9) vs decomposed e + combining accent
        decomposed = "e\u0301"  # e + combining acute accent
        composed = "\u00e9"     # é precomposed
        assert unicode_normalize(decomposed) == composed

    def test_plain_ascii(self):
        assert unicode_normalize("hello") == "hello"


class TestLowercase:
    def test_uppercase(self):
        assert lowercase("NYANJA") == "nyanja"

    def test_mixed(self):
        assert lowercase("Tonga") == "tonga"

    def test_already_lower(self):
        assert lowercase("bemba") == "bemba"


class TestRemovePunctuation:
    def test_removes_commas_and_periods(self):
        result = remove_punctuation("hello, world.")
        assert "," not in result
        assert "." not in result

    def test_keeps_apostrophe_by_default(self):
        result = remove_punctuation("it's fine")
        assert "'" in result

    def test_removes_apostrophe_when_flagged(self):
        result = remove_punctuation("it's fine", keep_apostrophe=False)
        assert "'" not in result

    def test_no_punctuation(self):
        assert remove_punctuation("hello world") == "hello world"


class TestCollapseWhitespace:
    def test_multiple_spaces(self):
        assert collapse_whitespace("hello   world") == "hello world"

    def test_tabs_and_newlines(self):
        assert collapse_whitespace("hello\t\nworld") == "hello world"

    def test_strips(self):
        assert collapse_whitespace("  hello  ") == "hello"


class TestReplaceSpacesWithBoundary:
    def test_single_space(self):
        assert replace_spaces_with_boundary("hello world") == "hello|world"

    def test_multiple_spaces(self):
        assert replace_spaces_with_boundary("a b c") == "a|b|c"

    def test_no_spaces(self):
        assert replace_spaces_with_boundary("hello") == "hello"


class TestNormalizeTranscription:
    def test_full_pipeline(self):
        text = "  Hello, World!  "
        result = normalize_transcription(text)
        assert result == "hello world"

    def test_for_training_adds_boundary(self):
        result = normalize_transcription("hello world", for_training=True)
        assert result == "hello|world"

    def test_zambian_text(self):
        text = "Mwenye wako alipita ku Lusaka."
        result = normalize_transcription(text)
        assert result == result.lower()
        assert "." not in result

    def test_empty_string(self):
        assert normalize_transcription("") == ""


class TestNormalizeBatch:
    def test_batch(self):
        texts = ["Hello World", "NYANJA language"]
        results = normalize_batch(texts)
        assert results == ["hello world", "nyanja language"]

    def test_batch_for_training(self):
        texts = ["hello world"]
        results = normalize_batch(texts, for_training=True)
        assert results == ["hello|world"]

    def test_empty_batch(self):
        assert normalize_batch([]) == []


class TestPreprocessFunction:
    def test_batch_dict(self):
        batch = {"transcription": ["Hello World", "Nyanja"]}
        result = preprocess_function(batch)
        assert result["transcription"] == ["hello world", "nyanja"]

    def test_preserves_other_keys(self):
        batch = {"transcription": ["test"], "audio": ["audio_data"]}
        result = preprocess_function(batch)
        assert "audio" in result
        assert result["audio"] == ["audio_data"]

    def test_for_training(self):
        batch = {"transcription": ["hello world"]}
        result = preprocess_function(batch, for_training=True)
        assert result["transcription"] == ["hello|world"]
