# S5: the ratio guard dropped.
p = "src/forward/forward_sites.cpp"; s = open(p).read()
def sub(old, new, count=1, start=None, end=None):
    global s
    i = s.index(start) if start else 0
    j = s.index(end, i) if end else len(s)
    region = s[i:j]
    assert region.count(old) == count, (old, region.count(old))
    s = s[:i] + region.replace(old, new) + s[j:]
sub('		if (ratio[d] < 0 || ratio[d] > INT64_C(0xFFFFFFFF)) return false;\n', '		(void)ratio[d]; /* MUTANT */\n')
open(p, "w").write(s)
