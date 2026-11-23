"""Data preparation: fra.txt → 8,000 / 1,000 / 1,000 (French tokens, English tokens) pairs.

Pipeline:
    raw line  "Go.\tVa !\tCC-BY 2.0 ..."
      → split on TAB          English, French (the attribution column is not used)
      → normalize()           unify special spaces and quotes, lowercase
      → tokenize()            list of word and punctuation tokens
      → filter                both sides non-empty and at most 10 tokens each
      → deduplicate           keep the first qualifying pair for each French sentence
      → Random(42).shuffle
      → first 10,000 pairs    train 8,000 / val 1,000 / test 1,000
"""

import random
import re
import unicodedata

DATA_PATH = "data/fra.txt"
MAX_LEN = 10          # max tokens per side; punctuation counts, special tokens do not
NUM_PAIRS = 10_000
SPLIT_SIZES = (8_000, 1_000, 1_000)
SEED = 42

# French typography puts a narrow no-break space before ! ? : and the corpus mixes
# several look-alike spaces and quotes. Unify them so one word has one spelling.
SPECIAL_CHARS = {
    "\xa0": " ",      # NO-BREAK SPACE
    "\u202f": " ",    # NARROW NO-BREAK SPACE
    "\u2009": " ",    # THIN SPACE
    "\u200b": "",     # ZERO WIDTH SPACE
    "\xad": "",       # SOFT HYPHEN
    "\u2019": "'",    # right single quotation mark
    "\u2018": "'",    # left single quotation mark
}

# A run of word characters (letters incl. é ç à, digits) or a single punctuation mark.
TOKEN_RE = re.compile(r"\w+|[^\w\s]")


def normalize(s: str) -> str:
    """NFC → replace special characters → lowercase → strip.

    normalize("Je n’ai pas peur\u202f!") == "je n'ai pas peur !"
    """
    s = unicodedata.normalize("NFC", s)       # "e" + combining accent → "é"
    for old, new in SPECIAL_CHARS.items():
        s = s.replace(old, new)
    return s.lower().strip()


def tokenize(s: str) -> list[str]:
    """tokenize("Don't go!") == ["don", "'", "t", "go", "!"]"""
    return TOKEN_RE.findall(normalize(s))


def load_pairs(path: str = DATA_PATH) -> list[tuple[list[str], list[str]]]:
    """Read, tokenize, filter and deduplicate by French. Keeps the original file order."""
    pairs = []
    seen = set()

    with open(path, encoding="utf-8") as f:
        for line in f:
            en, fr, _source = line.rstrip("\n").split("\t")
            fr_tokens = tokenize(fr)          # column 2: French → model input
            en_tokens = tokenize(en)          # column 1: English → model target

            if not fr_tokens or not en_tokens:
                continue
            if len(fr_tokens) > MAX_LEN or len(en_tokens) > MAX_LEN:
                continue

            # Deduplicate after filtering, so a line that fails the filter
            # cannot take the slot of a later qualifying translation.
            key = " ".join(fr_tokens)
            if key in seen:
                continue
            seen.add(key)

            pairs.append((fr_tokens, en_tokens))

    return pairs


def split_pairs(pairs):
    """Shuffle with an independent Random(SEED), take NUM_PAIRS, split into train / val / test."""
    pairs = list(pairs)                        # shuffle is in place; don't mutate the caller's list
    random.Random(SEED).shuffle(pairs)
    pairs = pairs[:NUM_PAIRS]

    n_train, n_val, _ = SPLIT_SIZES
    train = pairs[:n_train]
    val = pairs[n_train:n_train + n_val]
    test = pairs[n_train + n_val:]
    return train, val, test


if __name__ == "__main__":
    pairs = load_pairs()
    train, val, test = split_pairs(pairs)
    print(f"{len(pairs)} qualifying pairs; train/val/test = {len(train)}/{len(val)}/{len(test)}")
    for fr, en in train[:5]:
        print(" ".join(fr), "  =>  ", " ".join(en))
