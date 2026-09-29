#!/usr/bin/env python3
"""engine_identity.py -- route P's engine-identity check for the H witness (plan rev 8; rev 9; rev 10).

Before H is read on an engine prefix:
  1. (rev 8) its provenance commit -- <prefix>/share/superslm-provenance/provenance.json "commit", written by the
     pin tool -- must equal the expected candidate commit;
  2. (rev 9; coverage-mutants-rev8 5d) the sha256 of the library the witness LINKS must equal the installed
     library's sha256 recorded by the pin step (installed-lib.sha256), so nothing was swapped in after the pin;
  3. (rev 10; adversary strike round 5 F1) the code of the library the witness links must be the REFERENCE ENGINE:
     its codeid.py code_sha256 must equal that of the engine the harness built from the candidate commit in a clean
     Release build (codeid.py build-reference), exactly as route E's E3 and Q1's B5 require. Check 2 says the
     library is the one the pin step installed; only check 3 says the pin step installed the right engine (a pin
     step with a stray launcher passed checks 1 and 2).
usage: engine_identity.py <engine prefix> <expected 40-hex commit> <library the witness links> <reference engine json>
       -> prints "ok" (exit 0) or the reason (exit 1). With no reference engine json the reading is refused.
"""
import hashlib, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "route_e"))
import codeid
pre = Path(sys.argv[1]); lib = Path(sys.argv[3]) if len(sys.argv) > 3 else pre / "lib" / "libsuperslm.a"
p = pre / "share" / "superslm-provenance" / "provenance.json"
try:
    c = json.loads(p.read_text()).get("commit", "")
except (OSError, ValueError):
    print("no provenance.json in the prefix"); sys.exit(1)
if c != sys.argv[2]:
    print(f"linked prefix is {c[:12] or '?'}, candidate is {sys.argv[2][:12]}"); sys.exit(1)
rec = pre / "share" / "superslm-provenance" / "installed-lib.sha256"
try:
    want = rec.read_text().split()[0]
except (OSError, IndexError):
    print("no installed-library sha256 recorded by the pin step"); sys.exit(1)
try:
    got = hashlib.sha256(lib.read_bytes()).hexdigest()
except OSError:
    print(f"the library the witness links is unreadable: {lib}"); sys.exit(1)
if got != want:
    print(f"the library the witness links (sha256 {got[:12]}) is not the pinned library (sha256 {want[:12]})"); sys.exit(1)
if len(sys.argv) < 5:
    print("no reference engine to compare the linked library with"); sys.exit(1)
try:
    ref = json.loads(Path(sys.argv[4]).read_text())
except (OSError, ValueError):
    print(f"the reference engine is unreadable: {sys.argv[4]}"); sys.exit(1)
ok, why = codeid.compare(codeid.digest(str(lib)), ref)
if not ok:
    print(f"the library the witness links is not the reference engine: {why}"); sys.exit(1)
print("ok"); sys.exit(0)
