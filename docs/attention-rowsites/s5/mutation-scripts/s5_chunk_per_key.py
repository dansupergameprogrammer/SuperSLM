# S5: the chunk loop keeps its per-key loop.
p = "src/forward/forward_sites.cpp"; s = open(p).read()
def sub(old, new, count=1, start=None, end=None):
    global s
    i = s.index(start) if start else 0
    j = s.index(end, i) if end else len(s)
    region = s[i:j]
    assert region.count(old) == count, (old, region.count(old))
    s = s[:i] + region.replace(old, new) + s[j:]
sub('''						QkQ31ScoreRow(q_rot.data() + h * head_dim, k_rows_base, ratio, head_dim, width,
						              scores.data());''', '''						for (size_t row = 0; row < width; ++row) /* MUTANT */
							scores[row] = QkQ31Score(q_rot.data() + h * head_dim,
							                         k_rows_base + row * head_dim, ratio, head_dim);''', 1, start='SslmForwardStatus RunLayerLoopChunkBatched(')
open(p, "w").write(s)
