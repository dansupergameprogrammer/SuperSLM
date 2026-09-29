# Extra (coverage vitality, shared code): both channel pads nonzero -- the limbs' padded channels carry w = 2^40 and the
# key pack's padded channels 127. Either pad alone is output-equivalent (a padded channel multiplies a zero from the
# other side); together they add 127 * 2^40 per padded channel. Only rows with head_dim % 4 != 0 reach them.
p = "src/forward/forward_sites.cpp"; s = open(p).read()
def sub(old, new):
    global s
    assert s.count(old) == 1, old
    s = s.replace(old, new)
sub('const int64_t w = d < head_dim ? static_cast<int64_t>(q[d]) * ratio[d] : 0;', 'const int64_t w = d < head_dim ? static_cast<int64_t>(q[d]) * ratio[d] : (INT64_C(1) << 40); /* MUTANT */')
sub('base[(d / 4) * stride + t * 4 + d % 4] = d < head_dim ? rows[t][d] : int16_t{0};', 'base[(d / 4) * stride + t * 4 + d % 4] = d < head_dim ? rows[t][d] : int16_t{127}; /* MUTANT */')
open(p, "w").write(s)
