"""P4.1 tests leg (ruling 29, MRX review N3): per-suite check counts for superslm_tests.

  p41_count_suites.py instrument TEST_MAIN.cpp OUT.cpp
      Writes a copy of tests/test_main.cpp whose main() prints, after each top-level suite call (a statement
      `Name(...);` on its own line at main's first indentation, Run*Cells(GChecks, GFailures) and the Test* calls
      alike), one line `p41 suite <Name> <checks it added>`. Nothing else changes: the same calls, in the same order,
      add the same checks. The copy is compiled with -iquote <tree>/tests so its quoted includes resolve as before.
  p41_count_suites.py tally LOG
      Prints `<Name> <checks>` per suite (a name called more than once is summed), then `(unattributed) <n>`: the
      total minus the suites' sum (checks added outside a top-level call). Exit 1 when the log has no totals line.
"""
import re
import sys

CALL = re.compile(r"^\t([A-Za-z_]\w*)\(([^;]*)\);\s*(//.*)?$")
TOTAL = re.compile(r"^superslm tests: (\d+) checks")
SUITE = re.compile(r"^p41 suite (\S+) (-?\d+)\s*$")


def instrument(src_path, out_path):
    lines = open(src_path, encoding="utf-8", errors="surrogateescape").read().split("\n")
    out, in_main, n = [], False, 0
    for line in lines:
        if line.startswith("int main("):
            in_main = True
        elif in_main and line == "}":
            in_main = False
        m = CALL.match(line) if in_main else None
        if m and m.group(1) not in ("return",) and not m.group(1).startswith("std"):
            name = m.group(1)
            out.append('\t{ const int p41c = GChecks; %s(%s); std::printf("p41 suite %s %%d\\n", GChecks - p41c); }' % (name, m.group(2), name))
            n += 1
        else:
            out.append(line)
    if n == 0:
        sys.exit("p41_count_suites: no top-level call found in main() of %s" % src_path)
    open(out_path, "w", encoding="utf-8", errors="surrogateescape").write("\n".join(out))
    print("instrumented %d calls" % n)


def tally(log_path):
    suites, total = {}, None
    for line in open(log_path, encoding="utf-8", errors="replace"):
        line = line.rstrip("\r\n")
        m = SUITE.match(line)
        if m:
            suites[m.group(1)] = suites.get(m.group(1), 0) + int(m.group(2))
            continue
        t = TOTAL.match(line)
        if t:
            total = int(t.group(1))
    if total is None:
        sys.exit("p41_count_suites: no totals line in %s" % log_path)
    for name in sorted(suites):
        print("%s %d" % (name, suites[name]))
    print("(unattributed) %d" % (total - sum(suites.values())))


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "instrument":
        instrument(sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 3 and sys.argv[1] == "tally":
        tally(sys.argv[2])
    else:
        sys.exit(__doc__)
