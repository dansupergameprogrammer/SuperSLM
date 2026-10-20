"""T-3103 reserved-vocabulary paradigm expansion and canonical-form matching (plan rev 15 s5.3,
D-SLM8154, CA14-1). Eval-side instrument code. It names no reserved word: the reserved list is a sealed
input.

expand(lemma) is the lemma's full paradigm: regular noun and verb inflections by rule (plural -s, -es,
-ies; verb -s, -ed, -ing with e-drop, consonant doubling and y to i), irregular forms and irregular plurals
from the frozen lexicon, and every spelling variant with its own paradigm.
fold(token) is every base form a token could inflect from, under the same rules.

A text token matches a reserved lemma when, after C and clitic removal, the token, any of its folds or any
component of a hyphenated or closed compound is in the lemma's expanded paradigm.
"""
from __future__ import annotations

import json
import re
import sys
import unicodedata
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import canon  # noqa: E402

LEXICON = json.loads((HERE / "morph-lexicon.json").read_text(encoding="utf-8"))
_VOWELS = set("aeiou")
_WORD = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*", re.UNICODE)
MIN_COMPONENT = 4


def _cvc(w):
    return len(w) >= 3 and w[-1] not in _VOWELS | {"w", "x", "y"} and w[-2] in _VOWELS and w[-3] not in _VOWELS


def regular_forms(w):
    forms = {w}
    if w.endswith("y") and len(w) > 1 and w[-2] not in _VOWELS:
        forms |= {w[:-1] + "ies", w[:-1] + "ied"}
    elif w.endswith(("s", "x", "z", "ch", "sh")):
        forms.add(w + "es")
    else:
        forms.add(w + "s")
    if w.endswith("e"):
        forms |= {w + "d", w[:-1] + "ing"}
    else:
        forms |= {w + "ed", w + "ing"}
        if _cvc(w):
            forms |= {w + w[-1] + "ed", w + w[-1] + "ing"}
    if w.endswith("ie"):
        forms.add(w[:-2] + "ying")
    return forms


def _variants(w):
    out = {w}
    for a, b in LEXICON["spelling_variants"]:
        if w in (a, b):
            out |= {a, b}
    return out


@lru_cache(maxsize=None)
def expand(lemma: str) -> frozenset:
    lemma = unicodedata.normalize("NFC", lemma).lower()
    forms = set()
    for v in _variants(lemma):
        forms |= regular_forms(v)
        forms |= set(LEXICON["irregular_verbs"].get(v, []))
        forms |= set(LEXICON["irregular_plurals"].get(v, []))
    for f in list(forms):
        forms |= _variants(f)
    return frozenset(forms)


def fold(token: str) -> set:
    """Candidate base forms of one lower-case token."""
    t = token
    out = {t}
    for suf in ("ies", "ied"):
        if t.endswith(suf) and len(t) > 4:
            out.add(t[:-3] + "y")
    for suf in ("es", "s", "ed", "d", "ing"):
        if t.endswith(suf) and len(t) > len(suf) + 2:
            base = t[:-len(suf)]
            out |= {base, base + "e"}
            if len(base) >= 2 and base[-1] == base[-2]:
                out.add(base[:-1])
    return out


def tokens(text: str):
    s = canon.canon(text, "entity") if text != "-" else ""
    return [w for w in _WORD.findall(s)]


def components(token: str):
    parts = set(re.split(r"[-‐‑]", token))
    parts.add(token)
    return {p for p in parts if p}


class ReservedIndex:
    def __init__(self, lemmas):
        self.forms = {}
        for lemma in lemmas:
            for f in expand(lemma):
                self.forms.setdefault(f, lemma)

    def match(self, text: str):
        """The reserved lemmas a text uses (names stay in-process; callers report only refusals)."""
        hits = set()
        for tok in tokens(text):
            for comp in components(tok):
                for cand in fold(comp) | {comp}:
                    if cand in self.forms:
                        hits.add(self.forms[cand])
                # closed compounds: a reserved form of at least MIN_COMPONENT letters as a component
                for form, lemma in self.forms.items():
                    if len(form) >= MIN_COMPONENT and form != comp and (comp.startswith(form) or comp.endswith(form)):
                        rest = comp[len(form):] if comp.startswith(form) else comp[:-len(form)]
                        if len(rest) >= 3:
                            hits.add(lemma)
        return hits
