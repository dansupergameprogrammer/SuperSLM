# Extra, shared packer: the scalar channel tail (head_dim % 16) not packed (the stale pack is used).
p = "src/forward/forward_sites.cpp"; s = open(p).read()
def sub(old, new, count=1, start=None, end=None):
    global s
    i = s.index(start) if start else 0
    j = s.index(end, i) if end else len(s)
    region = s[i:j]
    assert region.count(old) == count, (old, region.count(old))
    s = s[:i] + region.replace(old, new) + s[j:]
sub('		for (size_t d = full16; d < nq * 4; ++d)', '		for (size_t d = nq * 4; d < nq * 4; ++d) /* MUTANT */')
open(p, "w").write(s)
