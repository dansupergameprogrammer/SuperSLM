"""T1a (debunker): the 10.3 timing instrument's must-reject, a restore that runs a prefill of the blob's length.

usage: python -I make_mutant.py <source root of a scratch copy>

Inserted in sslm_prefix_restore right after the blob validates and before its own draw: a scratch prefix of the
blob's `length` is begun on the same pool, prefilled in 64-token chunks (the timing cell's own chunk) with a fixed
token, and released. The restore's result is unchanged; only its cost becomes a prefill's. The 1,000x bar must refuse
it (expected ratio about 1x). Exact single-occurrence replacement; refuses a missing or repeated anchor.
"""
import sys

ANCHOR = "\tconst int64_t length = f.header.length;\n"
MUTANT = ANCHOR + """\t{  // MUTANT T1-restore-prefills (E4 T1a must-reject): the restore runs a prefill of the blob's length
\t\tsslm_prefix scratch = nullptr;
\t\tif (sslm_prefix_begin_budgeted(model, pool, static_cast<int32_t>(length), &scratch) == SSLM_OK && scratch) {
\t\t\tstd::vector<int32_t> ids(static_cast<size_t>(length), 220);
\t\t\tsize_t at = 0;
\t\t\twhile (at < ids.size()) {
\t\t\t\tint32_t consumed = 0;
\t\t\t\tif (sslm_prefix_prefill(model, scratch, ids.data() + at, static_cast<int32_t>(ids.size() - at), 64,
\t\t\t\t                        SSLM_SPAN_PROMPT, nullptr, &consumed) != SSLM_OK || consumed <= 0) break;
\t\t\t\tat += static_cast<size_t>(consumed);
\t\t\t}
\t\t\tsslm_prefix_release(scratch);
\t\t\tstd::fprintf(stderr, "MUTANT T1-restore-prefills: prefilled %zu of %lld\\n", at, static_cast<long long>(length));
\t\t}
\t}
"""

path = sys.argv[1] + "/src/sslm_abi.cpp"
src = open(path, encoding="utf-8", newline="").read()
eol = "\r\n" if "\r\n" in src else "\n"   # the box checks out with core.autocrlf=true
ANCHOR, MUTANT = ANCHOR.replace("\n", eol), MUTANT.replace("\n", eol)
n = src.count(ANCHOR)
if n != 1:
    raise SystemExit(f"anchor found {n} times, refusing")
src = src.replace(ANCHOR, MUTANT)
if "#include <cstdio>" not in src:
    src = src.replace("#include <cstring>" + eol, "#include <cstring>" + eol + "#include <cstdio>" + eol, 1)
open(path, "w", encoding="utf-8", newline="").write(src)
print("applied T1-restore-prefills to", path)
