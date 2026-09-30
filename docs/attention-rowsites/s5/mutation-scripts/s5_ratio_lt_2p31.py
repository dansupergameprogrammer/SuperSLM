# S5: the ratio guard -> < 2^31.
p = "src/forward/forward_sites.cpp"; s = open(p).read()
def sub(old, new, count=1, start=None, end=None):
    global s
    i = s.index(start) if start else 0
    j = s.index(end, i) if end else len(s)
    region = s[i:j]
    assert region.count(old) == count, (old, region.count(old))
    s = s[:i] + region.replace(old, new) + s[j:]
sub('ratio[d] > INT64_C(0xFFFFFFFF)', 'ratio[d] > INT64_C(0x7FFFFFFF) /* MUTANT */')
open(p, "w").write(s)
