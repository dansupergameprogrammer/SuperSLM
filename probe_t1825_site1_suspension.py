# T-1825 Loki probe: site 1 (embed) grouped funnel vs the committed-state interface.
#
# Executes the T-1822 design's own pinned rules (S4.1 grouped producer, S4.2
# exact shift-alignment consumer) on a concrete embed-shaped row, then routes
# the result through the engine's real embed->layer-0 interface:
# SequenceLayerState = { int8 hidden_codes[1536], ONE CarriedScale, layer_index }
# (forward_sites.h:441-449; commit at forward_sites.cpp:1673-1674 and :1508-1510;
# layer-0 consumer reads seq.hidden_codes at :1192).
#
# The design (S9.3) classifies M1's k[] as "transient (working rows)" and S12's
# lifetime row asserts "no committed state is added at any in-scope
# configuration". This probe measures what the layer-0 RMSNorm consumer
# reconstructs when the k[] sidecar does not cross the interface, against what
# S4.2 alignment reconstructs when it does.
#
# Disposable. Never merged. Reproduce: python probe_t1825_site1_suspension.py
import random

N = 1536          # hidden_size (Qwen2.5-1.5B)
G = 32            # design S4.1, swept set {32,128,256}; stage-C arm "M1-only all-sites at G=32"
K_CAP = 3         # design S6.1 row 1: site 1 k_cap = 3 (norm consumer, S6.3b)

# --- deterministic embed-shaped wide row -------------------------------------
# Magnitude profile in the measured cell: decile ratio inside 46.5-85.3x
# (design fact 2.3) with a dominant-channel shape (fact 2.9's rank0/median
# 22.7x at the outlier-bearing sites). Lognormal sigma=1.6 gives a 90/10
# percentile ratio near 60x; one channel planted at ~20x the bulk max.
# Seeded, reproducible.
rng = random.Random(18250807)
x = []
for i in range(N):
    mag = int(4096 * (2.718281828 ** rng.gauss(0.0, 1.6)))
    sign = -1 if rng.random() < 0.5 else 1
    x.append(sign * max(mag, 1))
bulk_max = max(abs(v) for v in x)
x[940] = 20 * bulk_max  # the massive-activation channel (fact 2.4's shape)

absx = sorted(abs(v) for v in x)
dec_lo = absx[len(absx) // 10]
dec_hi = absx[(9 * len(absx)) // 10]
print(f"row: n={N}, max|x|={max(absx)}, decile ratio hi/lo = {dec_hi / dec_lo:.1f}")

# --- S4.1: the grouped producer, exactly as pinned ---------------------------
# step 1: per-group maxima, row max identical to today's single reduction
groups = [x[g * G:(g + 1) * G] for g in range(N // G)]
Dg = [max(max(abs(v) for v in grp), 1) for grp in groups]
D = max(Dg)
assert D == max(absx)  # C29 input unchanged -- design step 1's claim, holds

# step 2: k_g = largest k in [0, k_cap] with (D'_g << k) <= D'
kg = []
for dg in Dg:
    k = 0
    while k < K_CAP and (dg << (k + 1)) <= D:
        k += 1
    kg.append(k)

# step 3: q_i = C22 composite on the pre-shifted operand, value-level:
# round-half-away-from-zero(x_shifted * 127 / D'), design's "no new rounding
# rule" claim. Integer-exact half-away form.
def c22_code(v, d):
    a = abs(v) * 127
    q = (2 * a + d) // (2 * d)
    return q if v >= 0 else -q

codes = []
for gi, grp in enumerate(groups):
    for v in grp:
        vs = v << kg[gi]
        assert abs(vs) <= D  # int64-exact shift claim: |x|<=D'_g, D'_g*2^kg<=D'
        q = c22_code(vs, D)
        assert abs(q) <= 127  # "codes stay in [-127,127] by k_g's own construction" -- holds
        codes.append(q)

refined = sum(1 for k in kg if k > 0)
print(f"producer: groups={len(kg)}, refined (k_g>0): {refined}/{len(kg)}"
      f" ({100.0 * refined / len(kg):.0f}%), k_g histogram "
      f"{ {k: kg.count(k) for k in sorted(set(kg))} }")

# element true value on the row grid: q_i * s_row * 2^-kg ; s_row = D/127
s_row = D / 127.0

# --- the committed-state interface (the engine's, verbatim) ------------------
# forward_sites.cpp:1673-1674: seq.hidden_codes[i] = embed_codes[i];
#                              seq.hidden_scale = embed_scale;
# forward_sites.h:441-449: int8_t* hidden_codes; CarriedScale hidden_scale;
#                          uint32_t layer_index;  -- no other row metadata.
# The design (S9.3) adds only site 18's peel records to this state; M1's k[]
# is classified transient. So what crosses is:
state_codes = codes[:]          # int8[1536]
state_scale = s_row             # exactly ONE CarriedScale
# k[] does NOT cross -- there is no field for it, by the design's own accounting.

# --- Path A: S4.2 alignment with k[] in hand (requires the sidecar) ----------
K = max(kg)
vA = []
for gi in range(len(kg)):
    for j in range(G):
        i = gi * G + j
        A = state_codes[i] << (K - kg[gi])          # exact left shift
        vA.append(A * state_scale / (1 << K))       # one grid, one scale
# S4.2 exactness claim: identical to q * s_row * 2^-kg -- verified:
for gi in range(len(kg)):
    for j in range(G):
        i = gi * G + j
        assert vA[i] == state_codes[i] * state_scale / (1 << kg[gi])

# --- Path B: the consumer reads the committed state as the state defines it --
# The committed contract (today's engine, and the design's S3 invariant): the
# row's value is code * scale. No k[] arrived; k_g is unknowable at this side.
vB = [state_codes[i] * state_scale for i in range(N)]

# --- the fracture, measured --------------------------------------------------
mis = [i for i in range(N) if vA[i] != vB[i]]
ratios = [vB[i] / vA[i] for i in mis if vA[i] != 0]
print(f"consumer: channels misread without k[]: {len(mis)}/{N} "
      f"({100.0 * len(mis) / N:.0f}%), misread factor 2^k_g up to "
      f"{max(ratios):.0f}x (mean {sum(ratios) / len(ratios):.2f}x)")

# layer-0 attn_norm (RMSNorm, gain=1, value level): y = v / rms(v), then the
# site's own funnel requantizes y on its max-abs grid. Compare output codes.
def rmsnorm_codes(v):
    rms = (sum(t * t for t in v) / len(v)) ** 0.5
    y = [t / rms for t in v]
    m = max(abs(t) for t in y)
    return [c22_code(int(round(t / m * (1 << 20))), 1 << 20) for t in y]

outA = rmsnorm_codes(vA)
outB = rmsnorm_codes(vB)
diff = sum(1 for a, b in zip(outA, outB) if a != b)
print(f"layer-0 attn_norm output codes differing (Path A vs Path B): {diff}/{N}")
print(f"=> the committed downstream state differs; gate D4's bit-exactness "
      f"(28x budget=1 vs 1x28) and D-SLM1287 slice invariance cannot both "
      f"hold with k[] transient at site 1.")

# --- the repair's own arithmetic (the design's storage claim, re-executed) ---
state_bytes = N + 16                       # codes + one CarriedScale (S9.3's own base)
peel = 40                                  # site 18, S9.3: +2.58%
k_sidecar = N // G                         # uint8 k[n/G] = 48 B at G=32
print(f"repair arithmetic: k[] must ride SequenceLayerState: +{k_sidecar} B "
      f"on {state_bytes} B = +{100.0 * k_sidecar / state_bytes:.2f}%; "
      f"with site-18 peel: +{peel + k_sidecar} B = "
      f"+{100.0 * (peel + k_sidecar) / state_bytes:.2f}% total "
      f"(design claims M1 adds nothing; peel-only +2.58%)")
