# S5: a2 by logical shift.
p = "src/forward/forward_sites.cpp"; s = open(p).read()
def sub(old, new, count=1, start=None, end=None):
    global s
    i = s.index(start) if start else 0
    j = s.index(end, i) if end else len(s)
    region = s[i:j]
    assert region.count(old) == count, (old, region.count(old))
    s = s[:i] + region.replace(old, new) + s[j:]
sub('w >> 30};', 'static_cast<int64_t>(static_cast<uint64_t>(w) >> 30)}; /* MUTANT */')
open(p, "w").write(s)
