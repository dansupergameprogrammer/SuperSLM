"""T-3103 sealed eval store: hash manifest and access log (plan rev 15 s5.3 "Data seal and access").

The store (D:\\_artifacts\\SuperSLM\\t3103\\sealed\\) holds the eval part's sealed content: reserved
vocabulary, writer commissions, written rows, annotations, adjudicated gold and split freezes. It is opened
only to eval-side readers, and to S0g and S4 decoding. Every access is a line in access-log.jsonl.

  seal    writes seal-manifest.json: SHA-256 of every file in the store (the access log and the manifest
          excluded), and prints only the manifest hash, file count and byte total.
  verify  re-hashes the store against the manifest. Exit 1 lists changed or missing paths, never content.
  access  appends {utc, seat, set, action, paths} to the access log. set must be 'eval-side'; anything else
          is recorded and returns exit 3: a seal breach under plan s0.1's re-gate rule.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path

EXCLUDE = {"access-log.jsonl", "seal-manifest.json"}


def files(store: Path):
    return sorted(p for p in store.rglob("*") if p.is_file() and p.name not in EXCLUDE)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("seal", "verify", "access"))
    ap.add_argument("--store", required=True)
    ap.add_argument("--seat")
    ap.add_argument("--set", dest="reader_set")
    ap.add_argument("--action")
    ap.add_argument("--paths", nargs="*", default=[])
    args = ap.parse_args(argv)
    store = Path(args.store)
    manifest_path = store / "seal-manifest.json"
    if args.cmd == "seal":
        manifest = {str(p.relative_to(store)).replace("\\", "/"): digest(p) for p in files(store)}
        data = (json.dumps({"sealed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                            "files": manifest}, indent=1, sort_keys=True) + "\n").encode()
        manifest_path.write_bytes(data)
        print(f"sealed files {len(manifest)} bytes {sum(p.stat().st_size for p in files(store))} "
              f"manifest_sha256 {hashlib.sha256(data).hexdigest()}")
        return 0
    if args.cmd == "verify":
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))["files"]
        now = {str(p.relative_to(store)).replace("\\", "/"): digest(p) for p in files(store)}
        bad = sorted(k for k in set(manifest) | set(now) if manifest.get(k) != now.get(k))
        for k in bad:
            print("CHANGED " + k)
        print(f"verified files {len(now)} changed {len(bad)}")
        return 1 if bad else 0
    entry = {"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), "seat": args.seat,
             "set": args.reader_set, "action": args.action, "paths": args.paths}
    with open(store / "access-log.jsonl", "a", encoding="utf-8", newline="\n") as log:
        log.write(json.dumps(entry, sort_keys=True) + "\n")
    if args.reader_set != "eval-side":
        print("SEAL BREACH: access by a seat outside the eval-side reader set was recorded")
        return 3
    print("access recorded")
    return 0


if __name__ == "__main__":
    sys.exit(main())
