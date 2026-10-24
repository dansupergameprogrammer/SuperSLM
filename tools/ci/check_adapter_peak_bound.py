"""Streamed adapters (plan rev 1.2 §3.1): holds the peak-bound census to the source.

Checks, from the repository root (or the path given):
  1. every term AdapterPeakTerms sums (src/adapter_core.h, the `uint64_t <field> = 0;   // PEAK-BOUND
     <term>:` rows) appears in docs/lora-residency/peak-bound-census.md's term table;
  2. every term has at least one `PEAK-BOUND <term>` tag at an allocation site outside the struct;
  3. every function the census's call-graph table names is defined or declared in the file it names.
Exit 0 when all hold; otherwise each failure is printed and the exit status is 1.
"""
import pathlib
import re
import sys

root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")
core = (root / "src/adapter_core.h").read_text()
census = (root / "docs/lora-residency/peak-bound-census.md").read_text()
failures = []

struct = re.search(r"struct AdapterPeakTerms \{(.*?)\n\};", core, re.S)
if not struct:
    sys.exit("AdapterPeakTerms not found in src/adapter_core.h")
terms = re.findall(r"//\s*PEAK-BOUND (\w+):", struct.group(1))
fields = re.findall(r"uint64_t (\w+) = 0;", struct.group(1))
if len(fields) != len(terms):
    failures.append(f"AdapterPeakTerms has {len(fields)} fields but {len(terms)} PEAK-BOUND term comments")
if len(terms) < 8:
    failures.append(f"only {len(terms)} terms found in AdapterPeakTerms")

sources = {}
for pattern in ("src/**/*.cpp", "src/**/*.h", "include/**/*.h"):
    for p in root.glob(pattern):
        sources[p.relative_to(root).as_posix()] = p.read_text(errors="replace")
outside = core.replace(struct.group(0), "")
for term in terms:
    if f"| `{term}` |" not in census:
        failures.append(f"term '{term}' is not in the census's term table")
    tagged = re.compile(rf"PEAK-BOUND {term}\b")
    hits = [f for f, text in sources.items() if tagged.search(text if f != "src/adapter_core.h" else outside)]
    if not hits:
        failures.append(f"term '{term}' has no PEAK-BOUND tag at any allocation site")

graph = census.split("## The call graph and its sites", 1)[-1]
for row in re.findall(r"^\| (.+?) \| `([^`]+)` \|", graph, re.M):
    names, path = row
    text = sources.get(path)
    if text is None:
        failures.append(f"census names {path}, which does not exist")
        continue
    for name in re.findall(r"`([A-Za-z_:]+)`", names):
        leaf = name.split("::")[-1]
        if not re.search(rf"\b{re.escape(leaf)}\s*\(", text) and not re.search(rf"\b{re.escape(leaf)}\b", text):
            failures.append(f"census names {name} in {path}, which it does not define")

for f in failures:
    print("check_adapter_peak_bound:", f)
print(f"check_adapter_peak_bound: {len(terms)} terms, {'FAIL' if failures else 'ok'}")
sys.exit(1 if failures else 0)
