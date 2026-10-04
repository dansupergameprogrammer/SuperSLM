"""T-3103 S0 grader canonicalizer C, frozen (plan rev 15 s5.1). Independent of adapter code.

For entity, state and value, and for each acceptable-span gold string, C applies these steps in order:
  1. Unicode NFC, then lowercase.
  2. Remove a possessive clitic ('s or ’s attached to a word).
  3. Trim, collapse internal whitespace, strip trailing punctuation.
  4. Strip one leading word from the closed list: the, a, an, that, this, these, those.
  5. For state only, strip one leading copula or auxiliary from the closed list: is, are, was, were, be,
     been, being, gets, get, got, becomes, become, has been, have been.
  6. Map the whole number words zero to twenty to digits, and percent or per cent to %.
Nothing else: no stemming, no synonym table, no stripping of possessive determiners. C is never applied
to the sentinel '-' and never runs before the semantic rules.
"""
from __future__ import annotations

import functools
import re
import unicodedata

DETERMINERS = ("the", "a", "an", "that", "this", "these", "those")
COPULAS = ("has been", "have been", "is", "are", "was", "were", "be", "been", "being", "gets", "get", "got",
           "becomes", "become")
NUMBERS = {w: str(i) for i, w in enumerate(
    ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
     "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty"))}
_CLITIC = re.compile(r"(?<=\w)['’]s\b")
SENTINEL = "-"


def _strip_trailing_punct(s: str) -> str:
    while s and unicodedata.category(s[-1]).startswith("P"):
        s = s[:-1].rstrip()
    return s


def _strip_leading(s: str, words) -> str:
    for w in words:                     # multi-word entries are listed first
        if s == w:
            return s                    # a lone determiner or copula is content, not a prefix
        if s.startswith(w + " "):
            return s[len(w) + 1:]
    return s


@functools.lru_cache(maxsize=None)
def canon(text: str, field: str) -> str:
    """field: 'entity', 'state' or 'value'."""
    if text == SENTINEL:
        return text
    s = unicodedata.normalize("NFC", text).lower()
    s = _CLITIC.sub("", s)
    s = " ".join(s.split())
    s = _strip_trailing_punct(s)
    s = _strip_leading(s, DETERMINERS)
    if field == "state":
        s = _strip_leading(s, COPULAS)
    s = re.sub(r"\bper cent\b", "%", s)
    s = re.sub(r"\bpercent\b", "%", s)
    s = " ".join(NUMBERS.get(w, w) for w in s.split(" "))
    return s
