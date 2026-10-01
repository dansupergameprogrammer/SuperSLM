"""Find a printable ASCII name whose FNV-1a 64 hash is 0, by lattice reduction.

With p = 1099511628211, H_i = p * (H_{i-1} ^ b_i). The XOR replaces the low byte L_{i-1} of
H_{i-1} by v_i = L_{i-1} ^ b_i, so H_i = p * (H_{i-1} - L_{i-1} + v_i) and
    H_n = p^n H_0 + sum_i p^(n-i+1) (v_i - L_{i-1})  (mod 2^64).
(H_{i-1} - L_{i-1}) is a multiple of 256, so L_i = 179 * v_i mod 256 (179 = p mod 256) depends
on v_i alone. H_n = 0 forces L_n = 0, hence v_n = 0 and b_n = L_{n-1}: the last byte is not
free. Writing L_i = 179 v_i - 256 k_i turns H_n = 0 into one congruence linear in the integer
unknowns (v_1..v_{n-1}, k_1..k_{n-1}) with box constraints: v_i in [64,127] and L_{i-1} in
[0,63] put b_i = L_{i-1} ^ v_i in [64,127] (DEL rejected), and L_{n-1} in [32,63] makes the last
byte printable. LLL once (all-integer, Cohen Alg. 2.6.7), then Babai nearest-plane per random
prefix until a solution lands inside the boxes; the result is verified by direct hashing.
"""
import pickle
import random
import sys
from fractions import Fraction
from pathlib import Path

P = 1099511628211
M64 = 1 << 64
OFFSET = 14695981039346656037


def fnv1a64(b: bytes) -> int:
    h = OFFSET
    for x in b:
        h = ((h ^ x) * P) & (M64 - 1)
    return h


def lll_integral(b, delta_num=99, delta_den=100):
    """Cohen, A Course in Computational Algebraic Number Theory, Alg. 2.6.7 (integral LLL)."""
    n = len(b)
    b = [list(r) for r in b]
    m = len(b[0])

    def dot(x, y):
        return sum(x[i] * y[i] for i in range(m))

    d = [0] * (n + 1)
    d[0] = 1
    lam = [[0] * n for _ in range(n)]
    k = 1
    kmax = 0
    d[1] = dot(b[0], b[0])

    def red(k, l):
        if 2 * abs(lam[k][l]) > d[l + 1]:
            q = (2 * lam[k][l] + d[l + 1]) // (2 * d[l + 1])
            b[k] = [b[k][i] - q * b[l][i] for i in range(m)]
            lam[k][l] -= q * d[l + 1]
            for i in range(l):
                lam[k][i] -= q * lam[l][i]

    def swap(k):
        b[k], b[k - 1] = b[k - 1], b[k]
        for j in range(k - 1):
            lam[k][j], lam[k - 1][j] = lam[k - 1][j], lam[k][j]
        lmb = lam[k][k - 1]
        B = (d[k - 1] * d[k + 1] + lmb * lmb) // d[k]
        for i in range(k + 1, kmax + 1):
            t = lam[i][k]
            lam[i][k] = (d[k + 1] * lam[i][k - 1] - lmb * t) // d[k]
            lam[i][k - 1] = (B * t + lmb * lam[i][k]) // d[k + 1]
        d[k] = B

    while k < n:
        if k > kmax:
            kmax = k
            for j in range(k + 1):
                u = dot(b[k], b[j])
                for i in range(j):
                    u = (d[i + 1] * u - lam[k][i] * lam[j][i]) // d[i]
                if j < k:
                    lam[k][j] = u
                else:
                    d[k + 1] = u
            if d[k + 1] == 0:
                raise ValueError("dependent rows")
        red(k, k - 1)
        if delta_den * d[k + 1] * d[k - 1] < delta_num * d[k] * d[k] - delta_den * lam[k][k - 1] ** 2:
            swap(k)
            k = max(1, k - 1)
        else:
            for l in range(k - 2, -1, -1):
                red(k, l)
            k += 1
    return b


class Babai:
    def __init__(self, B):
        self.B = B
        self.n = len(B)
        self.dim = len(B[0])
        self.Bs, self.norms = [], []
        for i in range(self.n):
            v = [Fraction(x) for x in B[i]]
            for j in range(i):
                mu = sum(Fraction(B[i][k]) * self.Bs[j][k] for k in range(self.dim)) / self.norms[j]
                v = [v[k] - mu * self.Bs[j][k] for k in range(self.dim)]
            self.Bs.append(v)
            self.norms.append(sum(x * x for x in v))

    def closest(self, target):
        t = [Fraction(x) for x in target]
        out = [0] * self.dim
        for i in range(self.n - 1, -1, -1):
            c = round(sum(t[k] * self.Bs[i][k] for k in range(self.dim)) / self.norms[i])
            if c:
                t = [t[k] - c * self.B[i][k] for k in range(self.dim)]
                out = [out[k] + c * self.B[i][k] for k in range(self.dim)]
        return out


K = 1 << 24
SV = 2          # v_i scaled by 2: box [64,127] -> radius 64 units, centre 191
SL = 2          # L_i (i <= n-2) scaled by 2: box [0,63] -> radius 64 units, centre 63
SLAST = 4       # L_{n-1} scaled by 4: box [32,63] -> radius 64 units, centre 190


def build_lattice(n):
    """Unknowns v_1..v_{n-1} (cols 1..n-1) and k_1..k_{n-1}; forms L_i in cols n..2n-2."""
    pw = [pow(P, e, M64) for e in range(n + 2)]
    cols = 1 + (n - 1) + (n - 1)
    rows = []
    for j in range(1, n):
        row = [0] * cols
        row[0] = K * ((pw[n - j + 1] - 179 * pw[n - j]) % M64)
        row[j] = SV
        row[n - 1 + j] = (SLAST if j == n - 1 else SL) * 179
        rows.append(row)
    for j in range(1, n):
        row = [0] * cols
        row[0] = K * ((256 * pw[n - j]) % M64)
        row[n - 1 + j] = -(SLAST if j == n - 1 else SL) * 256
        rows.append(row)
    row = [0] * cols
    row[0] = K * M64
    rows.append(row)
    return rows, pw


def solve(prefix, n, pw, bab):
    H0 = fnv1a64(prefix)
    L0 = H0 & 0xFF
    if L0 > 63:
        return None
    T = (pw[n] * (L0 - H0)) % M64
    target = [K * T] + [SV * 95 + 1] * (n - 1) + [SL * 31 + 1] * (n - 2) + [SLAST * 47 + 2]
    x = bab.closest(target)
    if x[0] != K * T:
        return None
    vs = [x[j] // SV for j in range(1, n)]
    Ls = [x[n - 1 + j] // SL for j in range(1, n - 1)] + [x[2 * n - 2] // SLAST]
    if any(not (64 <= v <= 127) for v in vs):
        return None
    if any(not (0 <= L <= 63) for L in Ls[:-1]) or not (32 <= Ls[-1] <= 63):
        return None
    out = bytearray(prefix)
    H = H0
    for v in vs + [0]:
        L = H & 0xFF
        b = L ^ v
        if not (32 <= b <= 126):
            return None
        out.append(b)
        H = (P * ((H - L) + v)) & (M64 - 1)
    return bytes(out) if fnv1a64(bytes(out)) == 0 else None


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 3030
    cache = Path(f"lll_last_n{n}.pkl")
    rows, pw = build_lattice(n)
    if cache.exists():
        R = pickle.loads(cache.read_bytes())
    else:
        R = lll_integral(rows)
        cache.write_bytes(pickle.dumps(R))
        print("LLL done", file=sys.stderr, flush=True)
    bab = Babai(R)
    rng = random.Random(seed)
    alphabet = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_"
    tries = 0
    while True:
        prefix = bytes(rng.choice(alphabet) for _ in range(5))
        tries += 1
        r = solve(prefix, n, pw, bab)
        if r is not None:
            print("n", n, "seed", seed, "tries", tries)
            print("name", r.decode("ascii"))
            print("hex", r.hex())
            print("fnv1a64", hex(fnv1a64(r)))
            print("printable", all(32 <= c <= 126 for c in r))
            return 0
        if tries % 200 == 0:
            print("tries", tries, file=sys.stderr, flush=True)


if __name__ == "__main__":
    sys.exit(main())
