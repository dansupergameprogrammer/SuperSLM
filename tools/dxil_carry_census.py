#!/usr/bin/env python3
"""DXIL census of SaturatingAdd64Gpu_'s carry-explicit form (engine 1.17.0, the RTX 50-series fix).

dxc at -O3 can re-fuse a split 32-bit add back into one native 64-bit add, and the Blackwell driver bug drops the
carry out of bit 31 of a native 64-bit add. This compiles a probe that calls the helper alone, and every engine
shader, with the shipping flags (cs_6_2, -O3, -HV 2018, -WX), then reports:

- probe: the number of carry-risk 64-bit adds in the helper alone. The carry-explicit form must give 0, with one
  carry pattern; the old native form gives 1 and 0. Anything else means this dxc re-fused the helper.
- per shader: the carry pattern count (an i32 add whose result is compared unsigned-less-than one of its operands)
  and the carry-risk i64 add count (an add whose operands can both carry out of bit 31 and whose high half is
  read). The carry-risk adds that remain are the engine's other 64-bit arithmetic, which the RTX 5060's full suite
  passes; the counts are for comparing dxc versions and the fix against the native form, not a pass rule.

Usage: dxil_carry_census.py --dxc <dxc> --shaders <src/gpu/shaders> --out <dir>
Exit 1 if the probe shows a re-fused add or any compile fails.
"""
import argparse, os, re, subprocess, sys

PROBE = '''#include "funnel_guards.hlsli"
RWStructuredBuffer<int64_t> buf : register(u0);
[numthreads(1,1,1)] void main(uint3 id : SV_DispatchThreadID)
{ buf[id.x*3+2] = SaturatingAdd64Gpu_(buf[id.x*3], buf[id.x*3+1]); }
'''
FLAGS = ['-T', 'cs_6_2', '-E', 'main', '-O3', '-HV', '2018', '-WX']
DEF = re.compile(r'^\s*(%[\w.]+) = (.*)$')
ADD64 = re.compile(r'^add (?:nuw |nsw )*i64 (%[\w.]+), (%[\w.]+)$')
ADD32 = re.compile(r'^\s*(%[\w.]+) = add (?:nuw |nsw )*i32 (%[\w.]+), (%[\w.]+)')
ULT32 = re.compile(r'icmp ult i32 (%[\w.]+), (%[\w.]+)')


def compile_one(dxc, src, inc, out_ll):
    r = subprocess.run([dxc] + FLAGS + ['-I', inc, '-Fc', out_ll, src], capture_output=True, text=True)
    return r.returncode == 0, r.stderr


def analyse(ll_path):
    defs, lines = {}, open(ll_path, encoding='utf-8', errors='replace').read().splitlines()
    for ln in lines:
        m = DEF.match(ln)
        if m:
            defs[m.group(1)] = m.group(2).split(' ;')[0].strip()

    def narrow(v, depth=0):
        d = defs.get(v, '')
        if depth > 4:
            return False
        if re.match(r'^(lshr i64 \S+, (3[2-9]|[4-6]\d))$', d) or re.match(r'^zext i(1|8|16|32) ', d):
            return True
        m = re.match(r'^add nuw nsw i64 (%[\w.]+), (%[\w.]+)$', d)
        return bool(m) and narrow(m.group(1), depth + 1) and narrow(m.group(2), depth + 1)

    def low_zero(v):
        d = defs.get(v, '')
        return bool(re.match(r'^shl (?:nuw |nsw )*i64 \S+, (3[2-9]|[4-6]\d)$', d)
                    or re.match(r'^and i64 \S+, -4294967296$', d))

    def low_only(v):
        # every use reads only the low 32 bits, which the bug leaves correct
        uses = [d for d in defs.values() if re.search(re.escape(v) + r'\b', d)]
        uses += [ln for ln in lines if not DEF.match(ln) and re.search(re.escape(v) + r'\b', ln)]
        return bool(uses) and all(re.match(r'^(trunc i64 %s to i32|and i64 %s, 4294967295)$' % (re.escape(v), re.escape(v)), u)
                                  for u in uses)

    full = 0
    for v, d in defs.items():
        m = ADD64.match(d)
        if m and not (narrow(m.group(1)) or narrow(m.group(2)) or low_zero(m.group(1)) or low_zero(m.group(2))
                      or low_only(v)):
            full += 1
    carries, adds = 0, {}
    for ln in lines:
        m = ADD32.match(ln)
        if m:
            adds[m.group(1)] = (m.group(2), m.group(3))
        m = ULT32.search(ln)
        if m and m.group(1) in adds and m.group(2) in adds[m.group(1)]:
            carries += 1
    return carries, full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dxc', required=True)
    ap.add_argument('--shaders', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ver = subprocess.run([a.dxc, '--version'], capture_output=True, text=True).stdout.strip()
    print(f'dxc: {ver}')
    bad = 0
    probe = os.path.join(a.out, 'probe_carry.hlsl')
    open(probe, 'w').write(PROBE)
    ok, err = compile_one(a.dxc, probe, a.shaders, os.path.join(a.out, 'probe_carry.ll'))
    if not ok:
        print('FAIL probe compile\n' + err); return 1
    pc, pf = analyse(os.path.join(a.out, 'probe_carry.ll'))
    verdict = 'OK' if (pf == 0 and pc >= 1) else 'RE-FUSED'
    bad += verdict != 'OK'
    print(f'probe SaturatingAdd64Gpu_: carry patterns {pc}, carry-risk i64 adds {pf} -> {verdict}')
    print(f'{"shader":32} {"carry":>6} {"carry_risk_i64_add":>19}')
    for f in sorted(os.listdir(a.shaders)):
        if not f.endswith('.hlsl'):
            continue
        n = f[:-5]
        ll = os.path.join(a.out, n + '.ll')
        ok, err = compile_one(a.dxc, os.path.join(a.shaders, f), a.shaders, ll)
        if not ok:
            print(f'{n:32} FAIL'); bad += 1; continue
        c, fw = analyse(ll)
        print(f'{n:32} {c:6} {fw:19}')
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
