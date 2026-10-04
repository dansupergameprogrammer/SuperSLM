"""T-3103 records check (plan rev 15 s5.3 and s7, D-SLM8161). An eval-side instrument that reports
counts and paths only. A hit is a seal breach under s0.1's re-gate rule.

It searches every tracked and untracked file under <root> (normally a project's Claude/ tree) and every
commit message in <repo> since <since> for sealed gate and held-out content:
  - each sealed utterance, exactly;
  - each sealed utterance after C-style normalization (NFC, casefold, punctuation stripped, whitespace
    collapsed);
  - any 8-word run of that normalized text, with actor and place names masked;
  - each gold string that is not a single common word (multi-word gold strings, and single words of the
    reserved vocabulary).

Usage:
  py -3 records_check.py --sealed <gate+held-out JSONL> [--reserved <sealed list>] --root <dir>
                         [--repo <git dir> --since <commit>]
Exit 0: no hits; 1: hits (counts and paths printed); 4: construction error.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_gates  # noqa: E402

TEXT_SUFFIXES = {".md", ".txt", ".json", ".jsonl", ".py", ".ps1", ".sh", ".cpp", ".h", ".csv", ".tsv", ".yaml",
                 ".yml", ".log", ".out", ""}


def build_needles(sealed_rows, reserved_lemmas=()):
    exact, norms, runs_, strings, names = set(), set(), set(), set(), set()
    reserved = {w.casefold() for w in reserved_lemmas}
    for r in sealed_rows:
        u = r.get("utterance", "")
        if not u:
            continue
        exact.add(u)
        norms.add(eval_gates.normalize(u))
        runs_ |= eval_gates.runs(eval_gates.normalize(u, eval_gates.names_of(r)))
        names |= {eval_gates.normalize(n) for n in eval_gates.names_of(r) if n}
        for g in eval_gates.gold_strings(r.get("document")):
            n = eval_gates.normalize(g)
            if len(n.split()) >= 2 or n in reserved:
                strings.add(n)
    names.discard("")
    pattern = re.compile(r"(?<!\w)(?:" + "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True))
                         + r")(?!\w)") if names else None
    return exact, norms, runs_, strings, pattern


def scan_text(text, needles):
    exact, norms, runs_, strings, pattern = needles
    kinds = Counter()
    if any(e in text for e in exact):
        kinds["exact"] += 1
    norm = eval_gates.normalize(text)
    padded = f" {norm} "
    if any(f" {n} " in padded for n in norms):
        kinds["normalized"] += 1
    masked = " ".join((pattern.sub(" NAME ", norm) if pattern else norm).split())
    masked_runs = eval_gates.runs(masked)
    if masked_runs & runs_:
        kinds["8-word-run"] += 1
    if any(f" {s} " in padded for s in strings):
        kinds["gold-string"] += 1
    return kinds


def files_under(root: Path):
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in TEXT_SUFFIXES and ".git" not in p.parts:
            yield p


def commit_messages(repo, since):
    out = subprocess.run(["git", "-C", str(repo), "log", "--format=%H%x00%B%x01", f"{since}..HEAD"],
                         capture_output=True, text=True, encoding="utf-8", check=True).stdout
    for chunk in out.split("\x01"):
        if "\x00" in chunk:
            sha, body = chunk.split("\x00", 1)
            yield sha.strip(), body


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sealed", required=True)
    ap.add_argument("--reserved")
    ap.add_argument("--root", required=True)
    ap.add_argument("--repo")
    ap.add_argument("--since")
    args = ap.parse_args(argv)
    try:
        sealed = [json.loads(l) for l in Path(args.sealed).read_text(encoding="utf-8").splitlines() if l.strip()]
        lemmas = []
        if args.reserved:
            r = json.loads(Path(args.reserved).read_text(encoding="utf-8"))
            lemmas = r["entity_lemmas"] + r["state_lemmas"]
        needles = build_needles(sealed, lemmas)
        hits = []
        n_files = 0
        for p in files_under(Path(args.root)):
            n_files += 1
            kinds = scan_text(p.read_text(encoding="utf-8", errors="replace"), needles)
            if kinds:
                hits.append((str(p), dict(kinds)))
        n_commits = 0
        if args.repo and args.since:
            for sha, body in commit_messages(args.repo, args.since):
                n_commits += 1
                kinds = scan_text(body, needles)
                if kinds:
                    hits.append((f"commit:{sha}", dict(kinds)))
    except Exception as error:  # noqa: BLE001
        print(f"CONSTRUCTION ERROR: {type(error).__name__}: {error}")
        return 4
    for where, kinds in hits:
        print("HIT " + where + " " + json.dumps(kinds, sort_keys=True))
    print(f"sealed rows {len(sealed)} files {n_files} commits {n_commits} hits {len(hits)}")
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main())
