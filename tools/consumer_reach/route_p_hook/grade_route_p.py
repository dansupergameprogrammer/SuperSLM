#!/usr/bin/env python3
"""grade_route_p.py -- grades the route P H witness output (plan rev 8), every reading against its expected value.

Expected values follow from how each engine is built, not from the witness:
  engine   accepts reserved          prefill calls run when                    hook bits by variant and k
  standin  0 only (v1.8.0, G33)      never                                     control/H1-H3: 0; Hplus_*: bit 0;
  optin    subsets of {bit 0}        bit 0 set, max_tasks >= 2, batch >= 8     Hbad_other_bit: bit 1;
  defon    subsets of {bit 0}        max_tasks >= 2, batch >= 8                Hk16...: bit 0 iff k >= 16;
                                                                               Hk_lt16: bit 0 iff k < 16
  wrong-engine, no-prov (rev 9, 5a: the prefix without provenance), lib-swap (rev 9, 5d: the default-on library
  installed under the opt-in prefix), pin-launcher (rev 10, adversary round 5 F1: the opt-in simulator pinned with a
  stray launcher forcing scalar): VOID (engine identity) on every reading. Dead counter: VOID wherever the install is accepted.
usage: run_route_p_hook_witness.sh ... | grade_route_p.py   (reads the witness output on stdin, echoes it, grades)
"""
import re, sys
bits = {"control": lambda k: 0, "H1_hex_zero": lambda k: 0, "H2_named_zero": lambda k: 0,
        "H3_reserved_deleted_Hook_value_init": lambda k: 0, "Hplus_prefill_bit": lambda k: 1,
        "Hplus_named_bit": lambda k: 1, "Hbad_other_bit": lambda k: 2,
        "Hk16_bit_only_at_k_ge_16": lambda k: 1 if k >= 16 else 0, "Hk_lt16": lambda k: 1 if k < 16 else 0}
def expect(variant, engine, k, dead):
    if engine in ("wrong-engine", "no-prov", "lib-swap", "pin-launcher"):   # rev 9: no-prov (5a), lib-swap (5d); rev 10: pin-launcher (F1)
        return "VOID (engine identity"
    b = bits[variant](k)
    accepted = (b == 0) if engine == "standin" else (b & ~1) == 0
    if not accepted:
        return "not hooked (engine refused"
    if dead:
        return "VOID (the finish never called run"
    hooked = {"standin": False, "sim-optin": bool(b & 1), "sim-defon": True}[engine]
    return "hooked" if hooked else "not hooked"
variant, n, bad = None, 0, []
for line in sys.stdin:
    sys.stdout.write(line)
    m = re.match(r"== (\S+)$", line.strip())
    if m and m.group(1) in bits:
        variant = m.group(1); continue
    m = re.match(r"\s*(?:(\S+) )?\s*\[([\w-]+)(, dead counter)?\] H\(k=(\d+)\) = (.*)$", line.rstrip())
    if not m:
        continue
    v = m.group(1) or variant; eng, dead, k, got = m.group(2), bool(m.group(3)), int(m.group(4)), m.group(5)
    want = expect(v, eng, k, dead); n += 1
    ok = got.startswith(want) and not (want == "hooked" and got != "hooked") and not (want == "not hooked" and got != "not hooked")
    if not ok:
        bad.append(f"{v} [{eng}{', dead' if dead else ''}] k={k}: expected '{want}...', got '{got}'")
print()
for b in bad:
    print("   UNEXPECTED:", b)
print(f"ROUTE P H: {'GREEN' if not bad and n else 'RED'} -- {n - len(bad)} of {n} readings as expected")
sys.exit(1 if bad or not n else 0)
