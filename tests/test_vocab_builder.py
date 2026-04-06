"""
Unit tests for src/utils/vocab_builder.py
"""

import json
import os
import tempfile

import pytest

from src.utils.vocab_builder import (
    SPECIAL_TOKENS,
    build_vocab,
    extract_characters,
    load_vocab,
    merge_vocabs,
    save_vocab,
)


class TestExtractCharacters:
    def test_basic(self):
        chars = extract_characters(["hello"])
        assert "h" in chars
        assert "e" in chars

    def test_space_excluded(self):
        chars = extract_characters(["hello world"])
        assert " " not in chars

    def test_lowercase(self):
        chars = extract_characters(["Hello"])
        assert "h" in chars
        assert "H" not in chars

    def test_multiple_transcriptions(self):
        chars = extract_characters(["abc", "def"])
        assert "a" in chars
        assert "d" in chars

    def test_empty(self):
        chars = extract_characters([])
        assert len(chars) == 0

    def test_deduplication(self):
        chars = extract_characters(["aaa"])
        assert chars == {"a"}


class TestBuildVocab:
    def test_special_tokens_present(self):
        vocab = build_vocab(["hello"])
        for token, idx in SPECIAL_TOKENS.items():
            assert token in vocab
            assert vocab[token] == idx

    def test_chars_included(self):
        vocab = build_vocab(["abc"])
        assert "a" in vocab
        assert "b" in vocab
        assert "c" in vocab

    def test_unique_ids(self):
        vocab = build_vocab(["hello world"])
        ids = list(vocab.values())
        assert len(ids) == len(set(ids))

    def test_with_extra_transcriptions(self):
        vocab = build_vocab(["abc"], extra_transcriptions=[["xyz"]])
        assert "x" in vocab
        assert "y" in vocab
        assert "z" in vocab

    def test_zambian_chars(self):
        texts = ["ndiyo", "tonga bana", "bemba mutende"]
        vocab = build_vocab(texts)
        assert "n" in vocab
        assert "t" in vocab
        assert "b" in vocab


class TestSaveAndLoadVocab:
    def test_round_trip(self):
        vocab = build_vocab(["nyanja tonga bemba"])
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "subdir", "vocab.json")
            save_vocab(vocab, path)
            assert os.path.exists(path)
            loaded = load_vocab(path)
            assert loaded == vocab

    def test_load_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            load_vocab("/nonexistent/path/vocab.json")

    def test_saved_as_valid_json(self):
        vocab = build_vocab(["hello"])
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "vocab.json")
            save_vocab(vocab, path)
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            assert isinstance(data, dict)


class TestMergeVocabs:
    def test_merges_chars(self):
        v1 = build_vocab(["nyanja"])
        v2 = build_vocab(["tonga"])
        merged = merge_vocabs([v1, v2])
        for char in "nyanja":
            assert char in merged
        for char in "tonga":
            assert char in merged

    def test_special_tokens_preserved(self):
        v1 = build_vocab(["abc"])
        v2 = build_vocab(["def"])
        merged = merge_vocabs([v1, v2])
        for token, idx in SPECIAL_TOKENS.items():
            assert merged[token] == idx

    def test_unique_ids(self):
        v1 = build_vocab(["abc"])
        v2 = build_vocab(["xyz"])
        merged = merge_vocabs([v1, v2])
        ids = list(merged.values())
        assert len(ids) == len(set(ids))

    def test_single_vocab(self):
        v1 = build_vocab(["hello"])
        merged = merge_vocabs([v1])
        assert merged == v1
