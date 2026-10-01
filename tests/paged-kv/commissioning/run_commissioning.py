#!/usr/bin/env python3
"""Paged-KV plan (rev 16.2) §8: the debunker's commissioning constructions, built and run.

One runner for the cloud (Linux, g++) and the box (Windows, MSVC, from run_commissioning.ps1, which
enters the developer shell first). README.md beside this file is the human-readable table; the
CONSTRUCTIONS list below is the executable one, and the two must agree.

What it does:
  build    exports this checkout's HEAD (tracked files only; a dirty src/include/tests/tools refuses)
           into <scratch>/<variant>/tree, for two variants:
             pristine  the tree as committed
             mutant    the tree plus mutants/debunk_mutants.patch (every mutation gated on
                       PKV_DEBUNK_MUTANT, so one library serves every mutant construction)
           appends the commissioning targets to the copy's tests/paged-kv/paged_kv.cmake (never to
           this checkout), configures CMake (Release, CPU only) and builds superslm_pkv_commission,
           superslm_pkv_commission_reftamper and superslm_pkv_c6. The tampered references and the
           tampered fixture are made from the copies, never in place.
  run      runs the constructions of one set and grades each by the instrument's own verdict: the
           runner's per-cell RED/green line and the FAIL reasons the instrument printed.

Exit status, matching instrument-commission.ps1 (non-zero = REJECTED):
  an *-accept set   0 when every construction was ACCEPTED by the instrument, else 1;
  an *-reject set   1 when every construction FIRED (rejected, for its stated reason), else 0 --
                    so a must-reject that was accepted, rejected for another reason, crashed, or
                    could not be built reads as accepted and the registry entry is written DEAD,
                    never a false COMMISSIONED;
  *-noresult        1 when the instrument reported "no result", else 0 (same fail-safe direction).
  --build-only / --list exit 0 on success, 2 on an infrastructure failure.
Run every set by hand once before -Commission: an infrastructure failure (a build that fails, a box
that is not quiet) is printed as such, but through the registry it can only read as DEAD or
REJECTS_HEALTHY.
"""

import argparse
import hashlib
import io
import os
import platform
import re
import shutil
import struct
import subprocess
import sys
import tarfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
IS_WIN = platform.system() == 'Windows'
EXE = '.exe' if IS_WIN else ''

# ---- the constructions ------------------------------------------------------------------------
#
# Each construction is one or more runs of one binary; the verdict is read from the last run. A run:
#   variant  pristine | mutant
#   binary   commission | commission_reftamper | c6
#   cells    exact cell ids (passed as "=<id>")
#   env      extra environment (PKV_DEBUNK_MUTANT, timing variables, ...)
#   fixtures default | tampered
# role: accept (every cell green, exit 0), reject (every cell RED with one of `reasons` in its FAIL
# lines), noresult (the output carries a "no result" verdict), observe (readings only, no verdict).

PROBE2_K = [2, 3, 16, 17, 255, 256, 257, 258, 511, 512, 513, 600]
PROBE1_K = [3, 4, 16, 17, 256, 257, 258, 513, 600]


def run(variant, binary, cells, env=None, fixtures='default'):
    return {'variant': variant, 'binary': binary, 'cells': cells, 'env': env or {}, 'fixtures': fixtures}


def C(cid, instrument, role, runs, what, reasons=(), platform_='any', timing=False, expect_note=''):
    return {'id': cid, 'instrument': instrument, 'role': role, 'runs': runs, 'what': what,
            'reasons': list(reasons), 'platform': platform_, 'timing': timing, 'note': expect_note}


ORACLE_ALL = ['CM.oracle:pkv_def', 'CM.oracle:pkv_qk', 'CM.oracle:pkv_odd', 'CM.oracle:pkv_32k']
PINS = ['CM.oracle.pin:persist', 'CM.oracle.pin:saturating']
COUNT_ALL = ['CM.count:n=1', 'CM.count:n=2', 'CM.count:n=3']


def probe2(kind, ks):
    return ['CM.probe2:%s:k=%d' % (kind, k) for k in ks]


def probe1(kind, ks):
    return ['CM.probe1:%s:k=%d' % (kind, k) for k in ks]


MUT = lambda m, **kw: dict({'PKV_DEBUNK_MUTANT': m}, **kw)  # noqa: E731
T79 = '7.9/C6'
T74 = '7.4/C6'
T79F = '7.9/C6:fixtures'
T74F = '7.4/C6:fixtures'
COMMISSIONED = {'SUPERSLM_PAGED_KV_TIMING_COMMISSIONED': '1'}

CONSTRUCTIONS = [
    # 1. byte-equality oracle ------------------------------------------------------------------
    C('oracle.A1', 'oracle', 'accept', [run('pristine', 'commission', ORACLE_ALL + PINS)],
      'unmutated paged build: every scenario on all four fixtures, and both pins replayed, equal v1.11.0'),
    C('oracle.A2', 'oracle', 'accept', [run('mutant', 'commission', ORACLE_ALL + PINS)],
      'the mutant library with PKV_DEBUNK_MUTANT unset (the mutant patch is inert when off)'),
    C('oracle.R1', 'oracle', 'reject', [run('mutant', 'commission', ORACLE_ALL, MUT('kv_flip'))],
      'one V byte (layer 1, KV head 1, position 17, d 0) lands XOR 1 in the engine landing',
      reasons=['K/V rows differ']),
    C('oracle.R2', 'oracle', 'reject', [run('mutant', 'commission', ['CM.oracle:pkv_def'], MUT('tok_flip'))],
      'one returned token is off by one (the step that leaves context_length 102); K/V untouched',
      reasons=['tokens differ']),
    C('oracle.R3', 'oracle', 'reject', [run('mutant', 'commission', ['CM.oracle:pkv_def'], MUT('blob_hdr'))],
      'every save flips one header bit (byte 100, forced_token_count); tokens and rows untouched',
      reasons=['blob differs']),
    C('oracle.R4', 'oracle', 'reject', [run('pristine', 'commission', ['CM.oracle:pkv_def'], fixtures='tampered')],
      'the fixture pkv_def.sslm with one weight byte changed (a host that builds different fixture bytes)',
      reasons=['was recorded against fixture']),
    C('oracle.R5', 'oracle', 'reject',
      [run('pristine', 'commission_reftamper', ['CM.oracle.one:pkv_def/lifecycle'])],
      'reference record "lifecycle prefill100+decode4" with one hex digit of rows= changed',
      reasons=['K/V rows differ']),
    C('oracle.R6', 'oracle', 'reject',
      [run('pristine', 'commission_reftamper', ['CM.oracle.one:pkv_qk/persist'])],
      'reference v1.11.0_pkv_qk.ref with one extra record "persist ghost" (a stage the run never reaches)',
      reasons=['records, reference has']),
    C('oracle.R7', 'oracle', 'reject', [run('pristine', 'commission_reftamper', ['CM.oracle.pin:persist'])],
      'the pinned blob v1.11.0_pkv_def_persist_saved with one K byte (layer 0, K, head 0, position 5, d 0) flipped',
      reasons=['K/V rows differ', 'blob differs', 'tokens differ']),

    # 2. legacy-create admission count ---------------------------------------------------------
    C('count.A1', 'count', 'accept', [run('pristine', 'commission', COUNT_ALL)],
      'exact states: n * 256 free at P counts n, n * 256 - 1 at P - 1 counts n - 1 (n = 1, 2, 3)'),
    C('count.A2', 'count', 'accept', [run('mutant', 'commission', COUNT_ALL)],
      'the mutant library with PKV_DEBUNK_MUTANT unset'),
    C('count.R1', 'count', 'reject', [run('mutant', 'commission', COUNT_ALL, MUT('leak1'))],
      'one-page deficit: the state\'s release leaks one page (shows at P)',
      reasons=['legacy-create count at P (']),
    C('count.R2', 'count', 'reject', [run('mutant', 'commission', COUNT_ALL, MUT('dfree1'))],
      'one-page excess: the state\'s release frees one page twice (shows at P - 1)',
      reasons=['legacy-create count at P - 1']),

    # 3a. two-sided fill probe ------------------------------------------------------------------
    C('probe2.A1', 'probe2', 'accept', [run('pristine', 'commission', probe2('exact', PROBE2_K))],
      'exact states, k in %s' % PROBE2_K),
    C('probe2.A2', 'probe2', 'accept', [run('mutant', 'commission', probe2('exact', PROBE2_K))],
      'the mutant library with PKV_DEBUNK_MUTANT unset'),
    C('probe2.R1', 'probe2', 'reject', [run('pristine', 'commission', probe2('deficit', PROBE2_K))],
      'claim k + 1 on a state with k free (one-page deficit against the claim)',
      reasons=['were not all admitted']),
    C('probe2.R2', 'probe2', 'reject', [run('pristine', 'commission', probe2('excess', PROBE2_K[1:]))],
      'claim k - 1 on a state with k free (one-page excess against the claim)',
      reasons=['were all admitted']),
    C('probe2.R3', 'probe2', 'reject', [run('mutant', 'commission', probe2('exact', PROBE2_K), MUT('leak1'))],
      'mutant: the state\'s release leaks one page (deficit)', reasons=['were not all admitted']),
    C('probe2.R4', 'probe2', 'reject', [run('mutant', 'commission', probe2('exact', PROBE2_K), MUT('dfree1'))],
      'mutant: the state\'s release frees one page twice (excess, a double free)', reasons=['were all admitted']),
    C('probe2.R5', 'probe2', 'reject',
      [run('mutant', 'commission', ['CM.probe2:excess:k=256', 'CM.probe2:excess:k=258'], MUT('budget_edge'))],
      'one-page excess (claim k - 1) on a build whose budgeted create also refuses budget >= cap - B for a '
      'reason other than capacity; the refusal leg\'s ceil(cap/B)-page create is then refused either way',
      reasons=['were all admitted']),

    # 3b. one-state fill probe -------------------------------------------------------------------
    C('probe1.A1', 'probe1', 'accept', [run('pristine', 'commission', probe1('exact', PROBE1_K))],
      'exact states, k in %s' % PROBE1_K),
    C('probe1.A2', 'probe1', 'accept', [run('mutant', 'commission', probe1('exact', PROBE1_K))],
      'the mutant library with PKV_DEBUNK_MUTANT unset'),
    C('probe1.R1', 'probe1', 'reject', [run('pristine', 'commission', probe1('deficit', PROBE1_K))],
      'claim k + 1 on a state with k free', reasons=['one-state fill probe at k=']),
    C('probe1.R2', 'probe1', 'reject', [run('pristine', 'commission', probe1('excess', PROBE1_K[1:]))],
      'claim k - 1 on a state with k free', reasons=['one-state fill probe at k=']),
    C('probe1.R3', 'probe1', 'reject', [run('mutant', 'commission', probe1('exact', PROBE1_K), MUT('leak1'))],
      'mutant: the state\'s release leaks one page', reasons=['one-state fill probe at k=']),
    C('probe1.R4', 'probe1', 'reject', [run('mutant', 'commission', probe1('exact', PROBE1_K), MUT('dfree1'))],
      'mutant: the state\'s release frees one page twice (a double free)', reasons=['one-state fill probe at k=']),
    C('probe1.R5', 'probe1', 'reject',
      [run('mutant', 'commission', ['CM.probe1:excess:k=256', 'CM.probe1:excess:k=258'], MUT('budget_edge'))],
      'as probe2.R5, for the one-state form', reasons=['one-state fill probe at k=']),

    # 4. timing harness: cloud twins (fixtures; noisy, readings recorded) -------------------------
    C('timing79.cloud.O1', 'timing79', 'observe', [run('pristine', 'c6', [T79F], COMMISSIONED)],
      'cloud twin, unmutated: pkv_def (B = 16) against pkv_odd (one page)', platform_='cloud', timing=True),
    C('timing79.cloud.R1', 'timing79', 'reject',
      [run('mutant', 'c6', [T79F], MUT('decode_slow', PKV_DEBUNK_DECODE_SLOW='0.25', **COMMISSIONED))],
      'cloud twin, paged decode slowed by a factor 1.25 (rate x 0.8: +20 points of slowdown)',
      reasons=['paged decode is'], platform_='cloud', timing=True),
    C('timing79.cloud.N1', 'timing79', 'noresult',
      [run('mutant', 'c6', [T79F], MUT('decode_slow', PKV_DEBUNK_DECODE_SLOW='0.01', **COMMISSIONED))],
      'cloud twin, paged decode slowed by 1% (below the run\'s resolving power)', platform_='cloud', timing=True),
    C('timing74.cloud.A1', 'timing74', 'accept',
      [run('pristine', 'c6', [T74F], {'SUPERSLM_PAGED_KV_74_MAX_RATIO': '2.0'})],
      'cloud twin, unmutated: reset and adopt at cap 32768 against cap 4096, bound 2.0', platform_='cloud',
      timing=True),
    C('timing74.cloud.R1', 'timing74', 'reject',
      [run('mutant', 'c6', [T74F], MUT('reset_capscale', SUPERSLM_PAGED_KV_74_MAX_RATIO='2.0'))],
      'cloud twin, reset made cap-proportional (writes cap x 64 bytes), bound 2.0',
      reasons=['reset: ratio'], platform_='cloud', timing=True),

    # 4. timing harness: the box (real artifacts; quiet box required) -----------------------------
    C('timing79.box.A1', 'timing79', 'accept',
      [run('pristine', 'c6', [T79], {'SUPERSLM_PAGED_KV_79_OUT': '@BASE'}),
       run('pristine', 'c6', [T79], dict({'SUPERSLM_PAGED_KV_79_BASELINE': '@BASE'}, **COMMISSIONED))],
      'unmutated pair: the paged build graded against a run of itself (A/A); must not be called a slowdown',
      platform_='box', timing=True),
    C('timing79.box.R1', 'timing79', 'reject',
      [run('pristine', 'c6', [T79], {'SUPERSLM_PAGED_KV_79_OUT': '@BASE'}),
       run('mutant', 'c6', [T79], MUT('decode_slow', PKV_DEBUNK_DECODE_SLOW='0.25',
                                      SUPERSLM_PAGED_KV_79_BASELINE='@BASE', **COMMISSIONED))],
      'paged decode slowed by a factor 1.25 (20% slower, 4x the bar), graded against the unmutated run',
      reasons=['paged decode is'], platform_='box', timing=True),
    C('timing79.box.N1', 'timing79', 'noresult',
      [run('pristine', 'c6', [T79], {'SUPERSLM_PAGED_KV_79_OUT': '@BASE'}),
       run('mutant', 'c6', [T79], MUT('decode_slow', PKV_DEBUNK_DECODE_SLOW='@SUBRES',
                                      SUPERSLM_PAGED_KV_79_BASELINE='@BASE', **COMMISSIONED))],
      'paged decode slowed by --subres-factor (default 0.01, ~1%), below the run\'s resolving power: must be '
      'reported as no result, neither a pass nor a fail', platform_='box', timing=True),
    C('timing74.box.A1', 'timing74', 'accept',
      [run('pristine', 'c6', [T74], {'SUPERSLM_PAGED_KV_74_MAX_RATIO': '@RATIO'})],
      'unmutated: reset and adopt on the 0.5B (cap 4096) against the 1.5B (cap 32768), bound --max74-ratio',
      platform_='box', timing=True),
    C('timing74.box.R1', 'timing74', 'reject',
      [run('mutant', 'c6', [T74], MUT('reset_capscale', SUPERSLM_PAGED_KV_74_MAX_RATIO='@RATIO'))],
      'reset made cap-proportional (writes cap x 64 bytes on every reset), bound --max74-ratio',
      reasons=['reset: ratio'], platform_='box', timing=True),
]

SETS = {}
for c in CONSTRUCTIONS:
    if c['role'] == 'observe':
        key = '%s-observe' % c['instrument']
    else:
        key = '%s-%s' % (c['instrument'], c['role'])
    if c['platform'] != 'any':
        key += '-' + c['platform']
    SETS.setdefault(key, []).append(c)

# ---- infrastructure ---------------------------------------------------------------------------


class Infra(Exception):
    pass


def sh(cmd, cwd=None, env=None, log=None, input_bytes=None):
    p = subprocess.run(cmd, cwd=cwd, env=env, input=input_bytes, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if log:
        with open(log, 'ab') as f:
            f.write(('$ %s\n' % ' '.join(cmd)).encode())
            f.write(p.stdout)
    return p.returncode, p.stdout


def head_commit(repo):
    code, out = sh(['git', '-C', repo, 'rev-parse', 'HEAD'])
    if code:
        raise Infra('git rev-parse HEAD failed in %s' % repo)
    return out.decode().strip()


def check_clean(repo):
    code, out = sh(['git', '-C', repo, 'status', '--porcelain', '--untracked-files=no', '--',
                    'src', 'include', 'tests/paged-kv', 'tools', 'CMakeLists.txt', 'cmake'])
    if code:
        raise Infra('git status failed in %s' % repo)
    if out.strip():
        raise Infra('the checkout has uncommitted changes under src/include/tests/tools; commit or stash them '
                    '(the constructions run on HEAD):\n' + out.decode())


CMAKE_APPEND = r'''
# ---- DEBUNK commissioning targets: appended to a SCRATCH COPY only by
# tests/paged-kv/commissioning/run_commissioning.py; never present in a checkout. ----
function(debunk_pkv_target name refdir)
	add_executable(${name} EXCLUDE_FROM_ALL ${PKV_DIR}/pkv_main.cpp ${ARGN})
	target_link_libraries(${name} PRIVATE superslm_test_injection)
	target_include_directories(${name} PRIVATE ${PKV_DIR} include src tests)
	target_compile_definitions(${name} PRIVATE PKV_REFERENCE_DIR="${refdir}")
	if(MSVC)
		target_compile_options(${name} PRIVATE /W4 /fp:precise)
	else()
		target_compile_options(${name} PRIVATE -Wall -Wextra -ffp-contract=off)
		find_package(Threads REQUIRED)
		target_link_libraries(${name} PRIVATE Threads::Threads)
	endif()
endfunction()
file(GLOB PKV_CM_SOURCES ${PKV_DIR}/commissioning/cm_*.cpp)
debunk_pkv_target(superslm_pkv_commission ${PKV_DIR}/reference ${PKV_CM_SOURCES})
debunk_pkv_target(superslm_pkv_commission_reftamper ${CMAKE_BINARY_DIR}/debunk_ref_tampered ${PKV_CM_SOURCES})
'''


def export_tree(repo, dest):
    code, data = sh(['git', '-C', repo, 'archive', '--format=tar', 'HEAD'])
    if code:
        raise Infra('git archive failed')
    if os.path.exists(dest):
        shutil.rmtree(dest)
    os.makedirs(dest)
    with tarfile.open(fileobj=io.BytesIO(data)) as t:
        t.extractall(dest)


def apply_patch(tree, patch, log):
    # The copy is made its own repository first, so `git apply` applies relative to it whatever
    # directory the scratch root sits in.
    for cmd in (['git', 'init', '-q'], ['git', 'apply', '--whitespace=nowarn', patch]):
        code, out = sh(cmd, cwd=tree, log=log)
        if code:
            raise Infra('%s failed in %s:\n%s' % (' '.join(cmd), tree, out.decode(errors='replace')))


# ---- ZRL (pkv_zrl.h), for the tampered pin ----------------------------------------------------

def zrl_decode(enc):
    assert enc[:4] == b'PZR1'
    size = struct.unpack_from('<Q', enc, 4)[0]
    out = bytearray()
    at = 12
    while len(out) < size:
        lit = struct.unpack_from('<I', enc, at)[0]
        at += 4
        out += enc[at:at + lit]
        at += lit
        zeros = struct.unpack_from('<I', enc, at)[0]
        at += 4
        out += bytes(zeros)
    assert len(out) == size and at == len(enc)
    return bytes(out)


def zrl_encode(data):
    # Literal / zero-run pairs; any valid split decodes the same (ZrlDecode reads the lengths).
    o = bytearray(b'PZR1') + struct.pack('<Q', len(data))
    at, n = 0, len(data)
    while at < n:
        lit_end = at
        while lit_end < n:
            if data[lit_end] == 0:
                z = lit_end
                while z < n and data[z] == 0 and z - lit_end < 16:
                    z += 1
                if z - lit_end >= 16 or z == n:
                    break
                lit_end = z
            else:
                lit_end += 1
        zero_end = lit_end
        while zero_end < n and data[zero_end] == 0:
            zero_end += 1
        o += struct.pack('<I', lit_end - at) + data[at:lit_end] + struct.pack('<I', zero_end - lit_end)
        at = zero_end
    return bytes(o)


def make_tampered_reference(tree, build):
    src = os.path.join(tree, 'tests', 'paged-kv', 'reference')
    dst = os.path.join(build, 'debunk_ref_tampered')
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    # R5: one hex digit of one rows= digest.
    p = os.path.join(dst, 'v1.11.0_pkv_def.ref')
    lines = open(p, 'rb').read().decode().split('\n')
    hit = 0
    for i, line in enumerate(lines):
        if line.startswith('lifecycle prefill100+decode4 '):
            m = re.search(r' rows=([0-9a-f]{64}) ', line)
            digest = m.group(1)
            flipped = digest[:-1] + ('0' if digest[-1] != '0' else '1')
            lines[i] = line.replace(' rows=' + digest + ' ', ' rows=' + flipped + ' ')
            hit += 1
    if hit != 1:
        raise Infra('reference tamper R5: record not found')
    open(p, 'wb').write('\n'.join(lines).encode())
    # R6: one extra record.
    p = os.path.join(dst, 'v1.11.0_pkv_qk.ref')
    text = open(p, 'rb').read().decode()
    if not text.endswith('\n'):
        text += '\n'
    text += 'persist ghost L=-1 sat=-1 blob=- rows=- tokens=1\n'
    open(p, 'wb').write(text.encode())
    # R7: one K byte of the persist pin, inside rows [0, L).
    p = os.path.join(dst, 'pins', 'v1.11.0_pkv_def_persist_saved.zrl')
    blob = bytearray(zrl_decode(open(p, 'rb').read()))
    assert blob[:4] == b'SSB5'
    L = struct.unpack_from('<q', blob, 60)[0]
    history = struct.unpack_from('<Q', blob, 112)[0]
    hidden = 192  # every pkv fixture (pkv_common.h HiddenSize)
    block = 156 + hidden + 4 * history + 4
    head_dim = 48
    pos = 5
    assert pos < L
    off = block + pos * head_dim  # layer 0, K, head 0, position 5, d 0 (flat layout, §3.1)
    blob[off] ^= 0x01
    enc = zrl_encode(bytes(blob))
    assert zrl_decode(enc) == bytes(blob)
    open(p, 'wb').write(enc)


def make_tampered_fixtures(fixtures, dest):
    if os.path.exists(dest):
        shutil.rmtree(dest)
    shutil.copytree(fixtures, dest)
    p = os.path.join(dest, 'pkv_def.sslm')
    data = bytearray(open(p, 'rb').read())
    # One byte an eighth of the way into the file (weight payload), low bit flipped, and the
    # artifact's own integrity SHA-256 (bytes 32..63, computed with those bytes as zero;
    # include/superslm/artifact.h) restamped: a validly built file that still maps, as a host
    # producing different fixture bytes would write it. Only its bytes, and so its file SHA-256,
    # differ from what the reference names. (Executed: on pkv_def this offset maps; some offsets in
    # the second half land in sections sslm_model_map validates and are refused, which would fail
    # the cell for a different reason.)
    off = len(data) * 5 // 40
    data[off] ^= 0x01
    h = hashlib.sha256()
    h.update(bytes(data[:32]))
    h.update(bytes(32))
    h.update(bytes(data[64:]))
    data[32:64] = h.digest()
    open(p, 'wb').write(bytes(data))
    return off


def build_variant(args, variant):
    vdir = os.path.join(args.scratch, variant)
    tree = os.path.join(vdir, 'tree')
    build = os.path.join(vdir, 'build')
    stamp = os.path.join(vdir, 'built-from')
    log = os.path.join(vdir, 'build.log')
    want = head_commit(args.repo) + ('+' + sha_file(args.patch) if variant == 'mutant' else '')
    have = open(stamp).read().strip() if os.path.exists(stamp) else ''
    if have == want and not args.rebuild:
        return
    check_clean(args.repo)
    os.makedirs(vdir, exist_ok=True)
    if os.path.exists(log):
        os.remove(log)
    if os.path.exists(stamp):
        os.remove(stamp)
    print('build %s: exporting HEAD into %s' % (variant, tree), flush=True)
    export_tree(args.repo, tree)
    if variant == 'mutant':
        apply_patch(tree, args.patch, log)
    with open(os.path.join(tree, 'tests', 'paged-kv', 'paged_kv.cmake'), 'a', newline='\n') as f:
        f.write(CMAKE_APPEND)
    if os.path.exists(build) and args.rebuild:
        shutil.rmtree(build)
    os.makedirs(build, exist_ok=True)
    if not os.path.exists(os.path.join(build, 'CMakeCache.txt')):
        cfg = ['cmake', '-S', tree, '-B', build, '-DCMAKE_BUILD_TYPE=Release', '-DSUPERSLM_BUILD_GPU=OFF']
        if shutil.which('ninja'):
            cfg[1:1] = ['-G', 'Ninja']
        code, out = sh(cfg, log=log)
        if code:
            raise Infra('configure of %s failed (see %s)' % (variant, log))
    make_tampered_reference(tree, build)
    print('build %s: building (log %s)' % (variant, log), flush=True)
    targets = ['superslm_pkv_commission', 'superslm_pkv_commission_reftamper', 'superslm_pkv_c6']
    code, out = sh(['cmake', '--build', build, '--target'] + targets + ['--parallel', str(args.jobs)], log=log)
    if code:
        raise Infra('build of %s failed (see %s)' % (variant, log))
    open(stamp, 'w').write(want + '\n')


def sha_file(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:16]


def exe_path(args, variant, binary):
    build = os.path.join(args.scratch, variant, 'build')
    name = {'commission': 'superslm_pkv_commission', 'commission_reftamper': 'superslm_pkv_commission_reftamper',
            'c6': 'superslm_pkv_c6'}[binary]
    for cand in (os.path.join(build, name + EXE), os.path.join(build, 'Release', name + EXE)):
        if os.path.exists(cand):
            return cand
    raise Infra('%s not built under %s' % (name, build))


# ---- quiet-box check (timing constructions) ------------------------------------------------------

BUSY_PROCESSES = ['UnrealEditor', 'UnrealEditor-Cmd', 'cl', 'link', 'dotnet', 'UnrealBuildTool', 'ninja',
                  'cmake', 'msbuild', 'cc1plus']


def idle_report():
    offenders = []
    load = ''
    if IS_WIN:
        if os.path.exists(r'D:\_ssu_build_lock'):
            offenders.append(r'D:\_ssu_build_lock present')
        code, out = sh(['tasklist', '/FO', 'CSV', '/NH'])
        names = set(re.findall(r'^"([^"]+)"', out.decode(errors='replace'), re.M))
        for b in BUSY_PROCESSES:
            if (b + '.exe') in names:
                offenders.append('process ' + b)
    else:
        load = '%.2f' % os.getloadavg()[0]
        code, out = sh(['ps', '-eo', 'comm='])
        names = set(out.decode(errors='replace').split())
        for b in BUSY_PROCESSES:
            if b in names:
                offenders.append('process ' + b)
    return offenders, load


# ---- running and grading ---------------------------------------------------------------------

CELL_LINE = re.compile(r'^\[[^\]]+\] (\S+) (RED|green) \((\d+) checks, (\d+) failed\)$', re.M)
FAIL_LINE = re.compile(r'^FAIL \[([^\]]+)\] (.*)$', re.M)
SUMMARY = re.compile(r'^(\d+) cells, (\d+) red; (\d+) checks, (\d+) failures$', re.M)
GRADE_79 = re.compile(r'^7\.9 (.*?): paged ([\d.]+) tok/s, one-page ([\d.]+) tok/s, slowdown (-?[\d.]+)%, resolving power ([\d.]+)%', re.M)
NO_RESULT = re.compile(r'no result', re.I)


def substitute(value, args, base):
    return value.replace('@BASE', base).replace('@SUBRES', args.subres_factor).replace('@RATIO', args.max74_ratio)


def run_one(args, c, r, idx, base):
    exe = exe_path(args, r['variant'], r['binary'])
    env = dict(os.environ)
    for k in ('PKV_DEBUNK_MUTANT', 'PKV_DEBUNK_DECODE_SLOW', 'SUPERSLM_PAGED_KV_79_OUT', 'SUPERSLM_PAGED_KV_79_BASELINE',
              'SUPERSLM_PAGED_KV_TIMING_COMMISSIONED', 'SUPERSLM_PAGED_KV_74_MAX_RATIO'):
        env.pop(k, None)
    if r['fixtures'] == 'tampered':
        env['SUPERSLM_PAGED_KV_FIXTURE_DIR'] = args.tampered_fixtures
    elif args.fixtures:
        env['SUPERSLM_PAGED_KV_FIXTURE_DIR'] = args.fixtures
    if args.artifacts:
        env['SUPERSLM_PAGED_KV_REAL_ARTIFACT_DIR'] = args.artifacts
    for k, v in r['env'].items():
        env[k] = substitute(v, args, base)
    cmd = [exe] + ['=' + cell for cell in r['cells']]
    t0 = time.time()
    p = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    secs = time.time() - t0
    out = p.stdout.decode(errors='replace')
    log = os.path.join(args.scratch, 'logs', '%s.run%d.log' % (c['id'], idx))
    os.makedirs(os.path.dirname(log), exist_ok=True)
    with open(log, 'w', encoding='utf-8') as f:
        f.write('$ %s\n' % ' '.join(cmd))
        f.write(''.join('  %s=%s\n' % (k, substitute(v, args, base)) for k, v in r['env'].items()))
        if r['fixtures'] == 'tampered':
            f.write('  SUPERSLM_PAGED_KV_FIXTURE_DIR=%s\n' % args.tampered_fixtures)
        f.write(out)
        f.write('\nexit %d after %.1f s\n' % (p.returncode, secs))
    return p.returncode, out, log, secs


def grade(c, code, out, args):
    cells = {m.group(1): m.group(2) for m in CELL_LINE.finditer(out)}
    fails = {}
    for m in FAIL_LINE.finditer(out):
        fails.setdefault(m.group(1), []).append(m.group(2))
    wanted = c['runs'][-1]['cells']
    ran = [cid for cid in wanted if cid in cells]
    missing = [cid for cid in wanted if cid not in cells]
    detail = []
    if c['role'] == 'accept':
        red = [cid for cid in ran if cells[cid] == 'RED']
        for cid in red:
            detail.append('%s RED: %s' % (cid, ' | '.join(fails.get(cid, ['(no FAIL line)'])[:3])))
        if missing:
            detail.append('cells that never reported: %s (exit %d)' % (missing, code))
        ok = code == 0 and not red and not missing and ran
        return ('ACCEPTED' if ok else 'REJECTED'), detail
    if c['role'] == 'reject':
        verdicts = []
        for cid in wanted:
            if cid not in cells:
                verdicts.append('NOT-REPORTED')
                detail.append('%s: never reported (exit %d: crash or abort, not the instrument\'s verdict)' % (cid, code))
            elif cells[cid] == 'green':
                verdicts.append('ACCEPTED')
                detail.append('%s: green -- the instrument ACCEPTED it' % cid)
            else:
                msgs = fails.get(cid, [])
                hit = [msg for msg in msgs if any(reason in msg for reason in c['reasons'])]
                if hit:
                    verdicts.append('FIRED')
                    detail.append('%s: RED: %s' % (cid, hit[0]))
                else:
                    verdicts.append('WRONG-REASON')
                    detail.append('%s: RED for another reason: %s' % (cid, ' | '.join(msgs[:3])))
        if all(v == 'FIRED' for v in verdicts):
            return 'FIRED', detail
        for v in ('ACCEPTED', 'WRONG-REASON', 'NOT-REPORTED'):
            if v in verdicts:
                return v, detail
    if c['role'] == 'noresult':
        g = GRADE_79.findall(out)
        for target, paged, one, slow, res in g:
            detail.append('%s: slowdown %s%%, resolving power %s%%' % (target, slow, res))
        injected = 100.0 * (1.0 - 1.0 / (1.0 + float(args.subres_factor)))
        if g and any(float(res) <= injected for (_, _, _, _, res) in g):
            detail.append('INCONCLUSIVE: a resolving power at or below the injected %.2f%%; rerun with a smaller '
                          '--subres-factor' % injected)
            return 'INCONCLUSIVE', detail
        if NO_RESULT.search(out):
            return 'NO-RESULT', detail
        red = [cid for cid in wanted if cells.get(cid) == 'RED']
        return ('REPORTED-FAIL' if red or code else 'REPORTED-PASS'), detail
    # observe
    for target, paged, one, slow, res in GRADE_79.findall(out):
        detail.append('%s: paged %s, one-page %s tok/s, slowdown %s%%, resolving power %s%%' % (target, paged, one, slow, res))
    for line in out.splitlines():
        if line.startswith('7.4 ') or line.startswith('7.9 '):
            if not GRADE_79.match(line):
                detail.append(line)
    red = [cid for cid in wanted if cells.get(cid) == 'RED']
    return ('OBSERVED (cell RED: %s)' % '; '.join(sum((fails.get(cid, []) for cid in red), [])[:2]) if red else 'OBSERVED (cell green)'), detail


def run_construction(args, c):
    if c['timing']:
        offenders, load = idle_report()
        if offenders and not args.allow_busy:
            return 'NOT-IDLE', ['box not quiet before the run: %s' % '; '.join(offenders)], []
    base = os.path.join(args.scratch, 'logs', c['id'] + '.baseline.txt')
    if os.path.exists(base):
        os.remove(base)
    logs = []
    code, out = 0, ''
    for i, r in enumerate(c['runs']):
        code, out, log, secs = run_one(args, c, r, i, base)
        logs.append('%s (%.0f s, exit %d)' % (log, secs, code))
        if i < len(c['runs']) - 1 and code != 0:
            return 'INFRA', ['an earlier run (baseline) exited %d; see %s' % (code, log)], logs
    verdict, detail = grade(c, code, out, args)
    if c['timing']:
        offenders, load = idle_report()
        detail.append('quiet after: %s%s' % ('yes' if not offenders else 'NO: ' + '; '.join(offenders),
                                             (' (load %s)' % load) if load else ''))
        if offenders and not args.allow_busy:
            verdict = 'NOT-IDLE (' + verdict + ')'
    return verdict, detail, logs


REQUIRED = {'accept': 'ACCEPTED', 'reject': 'FIRED', 'noresult': 'NO-RESULT'}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--repo', default=DEFAULT_REPO)
    ap.add_argument('--scratch', required=False)
    ap.add_argument('--fixtures', default=os.environ.get('SUPERSLM_PAGED_KV_FIXTURE_DIR', ''))
    ap.add_argument('--artifacts', default=os.environ.get('SUPERSLM_PAGED_KV_REAL_ARTIFACT_DIR', ''),
                    help='directory holding the two real artifacts under the names the C6 cells read (box)')
    ap.add_argument('--set', action='append', default=[], help='a set name (see --list); repeatable')
    ap.add_argument('--construction', action='append', default=[], help='one construction id; repeatable')
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--build-only', action='store_true')
    ap.add_argument('--rebuild', action='store_true')
    ap.add_argument('--jobs', type=int, default=os.cpu_count() or 4)
    ap.add_argument('--subres-factor', default='0.01')
    ap.add_argument('--max74-ratio', default='2.0')
    ap.add_argument('--allow-busy', action='store_true', help='run timing constructions on a busy machine (cloud twin)')
    args = ap.parse_args()
    args.patch = os.path.join(HERE, 'mutants', 'debunk_mutants.patch')

    if args.list:
        for name in sorted(SETS):
            print(name)
            for c in SETS[name]:
                print('    %-20s %s' % (c['id'], c['what']))
        return 0
    if not args.scratch:
        ap.error('--scratch is required')
    args.scratch = os.path.abspath(args.scratch)
    os.makedirs(args.scratch, exist_ok=True)

    chosen = []
    for s in args.set:
        if s not in SETS:
            ap.error('no set %s (see --list)' % s)
        chosen += SETS[s]
    by_id = {c['id']: c for c in CONSTRUCTIONS}
    for cid in args.construction:
        if cid not in by_id:
            ap.error('no construction %s (see --list)' % cid)
        chosen.append(by_id[cid])

    try:
        variants = sorted({r['variant'] for c in chosen for r in c['runs']}) if chosen else ['mutant', 'pristine']
        for v in variants:
            build_variant(args, v)
        if any(r['fixtures'] == 'tampered' for c in chosen for r in c['runs']):
            if not args.fixtures:
                raise Infra('--fixtures (or SUPERSLM_PAGED_KV_FIXTURE_DIR) is required')
            args.tampered_fixtures = os.path.join(args.scratch, 'fixtures-tampered')
            off = make_tampered_fixtures(args.fixtures, args.tampered_fixtures)
            print('tampered fixture: pkv_def.sslm byte %d XOR 1 in %s' % (off, args.tampered_fixtures))
    except Infra as e:
        print('INFRASTRUCTURE FAILURE: %s' % e)
        roles = {c['role'] for c in chosen}
        return 0 if roles and roles <= {'reject', 'noresult'} else 2
    if args.build_only or not chosen:
        return 0

    results = []
    for c in chosen:
        print('== %s [%s, must %s] %s' % (c['id'], c['instrument'], c['role'], c['what']), flush=True)
        try:
            verdict, detail, logs = run_construction(args, c)
        except Infra as e:
            verdict, detail, logs = 'INFRA', [str(e)], []
        for d in detail:
            print('   ' + d)
        for log in logs:
            print('   log ' + log)
        required = REQUIRED.get(c['role'])
        status = 'as required' if verdict == required else ('NOT AS REQUIRED' if required else 'readings only')
        print('   -> %s (%s)' % (verdict, status), flush=True)
        results.append((c, verdict, required))

    print('\nsummary:')
    for c, verdict, required in results:
        print('  %-20s %-9s %-14s required %s' % (c['id'], c['role'], verdict, required or '-'))
    roles = {c['role'] for c, _, _ in results}
    if roles == {'accept'}:
        return 0 if all(v == r for _, v, r in results) else 1
    if roles <= {'reject', 'noresult'}:
        return 1 if all(v == r for _, v, r in results) else 0
    # mixed or observe-only selections are for reading, not for the registry
    return 0


if __name__ == '__main__':
    sys.exit(main())
