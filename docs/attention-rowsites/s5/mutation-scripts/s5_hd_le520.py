# S5: head_dim <= 512 -> <= 520 (the constant, so the pack and limb buffers grow with it and only the guard moves).
p = "src/forward/forward_sites.cpp"; s = open(p).read()
def sub(old, new, count=1, start=None, end=None):
    global s
    i = s.index(start) if start else 0
    j = s.index(end, i) if end else len(s)
    region = s[i:j]
    assert region.count(old) == count, (old, region.count(old))
    s = s[:i] + region.replace(old, new) + s[j:]
sub('constexpr size_t kQ31RowMaxHeadDim = 512;', 'constexpr size_t kQ31RowMaxHeadDim = 520; /* MUTANT */')
open(p, "w").write(s)
