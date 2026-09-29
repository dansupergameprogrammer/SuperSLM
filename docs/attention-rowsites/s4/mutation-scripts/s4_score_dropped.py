# Guard: the score-magnitude conjunct dropped (output-equivalent).
p='src/intmath.cpp'; s=open(p).read()
def sub(old, new, count=1, start=None, end=None):
    global s
    i = s.index(start) if start else 0
    j = s.index(end, i) if end else len(s)
    region = s[i:j]
    assert region.count(old) == count, (old, region.count(old))
    s = s[:i] + region.replace(old, new) + s[j:]
sub('\t\tif (v > kSoftmaxFastScoreLimit || v < -kSoftmaxFastScoreLimit) return false;\n', '\t\t/* MUTANT */\n', start='bool SoftmaxFastGuard(', end='struct SoftmaxFastRow')
open(p,"w").write(s)
