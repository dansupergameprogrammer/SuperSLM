# S3 call site: the funnel keeps its element loop (the row leaf is never entered).
p='src/forward/checked_chain_funnel.cpp'; s=open(p).read()
def sub(old, new, count=1, start=None, end=None):
    global s
    i = s.index(start) if start else 0
    j = s.index(end, i) if end else len(s)
    region = s[i:j]
    assert region.count(old) == count, (old, region.count(old))
    s = s[:i] + region.replace(old, new) + s[j:]
sub('RequantRowWide(wide_row, n, preflight.reciprocal, preflight.normalized.s, out_codes);', 'for (size_t i = 0; i < n; ++i) out_codes[i] = RequantTokenCodeWide(wide_row[i], preflight.reciprocal, preflight.normalized.s); /* MUTANT */')
open(p,'w').write(s)
