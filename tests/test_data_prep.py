from pathlib import Path

import pytest

from data_prep import DATA_PATH, load_pairs, normalize, split_pairs, tokenize


def test_normalize_special_spaces_and_quotes():
    assert normalize("  Je n’ai pas peur !  ") == "je n'ai pas peur !"


def test_normalize_composes_accents():
    assert normalize("Café") == "café"


@pytest.mark.parametrize("text, tokens", [
    ("Don't go!", ["don", "'", "t", "go", "!"]),
    ("Est-ce « vrai »\xa0?", ["est", "-", "ce", "«", "vrai", "»", "?"]),
    ("   ", []),
])
def test_tokenize(text, tokens):
    assert tokenize(text) == tokens


needs_data = pytest.mark.skipif(not Path(DATA_PATH).exists(), reason="data/fra.txt not downloaded")


@pytest.fixture(scope="module")
def pairs():
    return load_pairs()


@needs_data
def test_filter_and_deduplicate(pairs):
    # Expected counts are for the 2026-02-13 release of fra.txt
    assert len(pairs) == 149_414
    assert pairs[0] == (["va", "!"], ["go", "."])
    assert sum(fr == ["va", "!"] for fr, _ in pairs) == 1
    assert all(0 < len(f) <= 10 and 0 < len(e) <= 10 for f, e in pairs)


@needs_data
def test_split_is_seeded_and_does_not_mutate(pairs):
    train, val, test = split_pairs(pairs)
    assert (len(train), len(val), len(test)) == (8_000, 1_000, 1_000)
    assert " ".join(train[0][0]) == "merci d ' avoir décoré ma maison !"
    assert " ".join(train[0][1]) == "thank you for decorating my house ."
    assert pairs[0] == (["va", "!"], ["go", "."])
