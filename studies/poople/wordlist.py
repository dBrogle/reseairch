"""Word-list loading for the Poople study.

Loads the ENABLE2k word list (downloading + caching it on first use) and
filters it down to the set of valid four-letter words used as graph nodes.
"""

import urllib.request

from studies.poople.config import WORD_LEN, WORDLIST_PATH, WORDLIST_URL


def _ensure_wordlist() -> None:
    """Download the raw word list to the local cache if it isn't there yet."""
    if WORDLIST_PATH.exists():
        return
    WORDLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading word list from {WORDLIST_URL} ...")
    urllib.request.urlretrieve(WORDLIST_URL, WORDLIST_PATH)
    print(f"  Cached to {WORDLIST_PATH}")


def load_words() -> set[str]:
    """Return the set of valid four-letter words (lowercase a–z only)."""
    _ensure_wordlist()
    words: set[str] = set()
    with open(WORDLIST_PATH) as f:
        for line in f:
            w = line.strip().lower()
            if len(w) == WORD_LEN and w.isalpha() and w.isascii():
                words.add(w)
    return words


def is_valid_word(word: str, words: set[str] | None = None) -> bool:
    """Whether `word` is a legal four-letter Poople word."""
    if words is None:
        words = load_words()
    return word in words
