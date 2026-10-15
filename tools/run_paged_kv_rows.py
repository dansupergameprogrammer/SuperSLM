"""Paged-KV plan rev 16.21 cell 7.9 (§23.8 step T4): the five-process 7.9 driver.

One model, one blob per row restored into both arms (a legacy holder in a page pool at the derived B, rotated,
and a legacy holder in the same build's contiguous pool option), 104 timed pairs per process after 4 warm-up
tokens, five processes on one boot. The process statistic is the median of s_i = tau_paged / tau_contiguous - 1;
the row rule: PASS when the largest of the five process medians is <= 0.05, FAIL when their median is > 0.05,
NO RESULT otherwise. A row is graded only on five complete processes; for a construction with a paged arm
(production, 7.9-MR-OPT), also only when their five measured-holder id lists agree and the scatter guard admits those
ids (M <= M99(n, P) at the n read after the run). The must-accept constructions MA-1 and MA-2 have no paged arm: their
row record reads `guard: "not applicable (no paged arm)"` and the row rule alone grades them (D-SLM8466 (1)).

Every refusal under the per-process rule has its own exit code, writes no record for the refused process, and leaves
the row ungraded: 79-6 a blob or manifest mismatch (checked before each process restores the blob), 79-7 a pair
executable that is not the construction it was run as (its identity line is not the construction's own, checked once
per process before any pair is read: a seam library, or another construction's build; an MA run that printed an IDS
page-id line or a RESERVE line; or a production or 7.9-MR-OPT run that printed no page ids or no RESERVE line), 79-8
a token mismatch between the arms, 79-9 a process with other than 104 timed pairs. A refused row writes no row record
at a fresh result path; where an earlier run's record already stands at that path, the refusal replaces it with a REFUSED
record carrying the exit code and no verdict, so a refused run never leaves an earlier verdict standing there (T-3202
F-12, D-SLM8461). Any other failure of the driver (an unparseable pair executable output, a driver fault) refuses
the same way with EXIT_INFRASTRUCTURE, never as a raw crash (T-3203 N-8). A process whose boot changed, or whose pair
executable crashed, is discarded and rerun whole. The box-global C6 timing lock (one format and protocol with the
other legs that take it, D-SLM8464; see timing_lock) is held from the first process until the row record is published.

Command lines (main):
    run_paged_kv_rows.py --pair-cmd JSON --manifest M --model F --restore N --slot S --result R
                         [--lock L] [--page-positions 64|16] [--construction C]
    run_paged_kv_rows.py --aggregate --slot R79A|R79MR --runs RECORD [RECORD ...] --result R
--page-positions selects the guard's bound table: 64 for the production rows, 16 for 7.9-MR-OPT's (pass the row as
--slot). --construction names what the row's pair command builds: production (the default), MA-1, MA-2 or 7.9-MR-OPT;
it is recorded in the row record, and 7.9-MR-OPT runs at --page-positions 16, every other construction at 64.

The aggregate mode (D-SLM8463) writes the slot record of R79A or R79MR, which run several constructions: one
result.json with `slot`, `runs` (each run's row, construction, status, verdict, exe_sha256 and record path),
`status` (STANDING only when every run is STANDING, REFUSED otherwise) and `exe_sha256`. It refuses, writing no
aggregate (and superseding an existing record at R as above), when the runs' exe_sha256 differ or when the runs are
not exactly the slot's set: R79A is MA-1 at R2b, R4a and R4b plus MA-2 at R4a; R79MR is 7.9-MR-OPT at R2b, R4a and
R4b.

Records (D-SLM8466 (2)): every process record carries `argv` (the exact pair command it ran) and `identity` (the
identity line it read); the row record carries `argv` (the driver's own command line) and `identity`. The label
--construction is checked against the identity line the pair executable prints, so a record shows which construction
ran without trusting the label.

The pair executable's contract, per construction (the identity strings are CONSTRUCTION_IDENTITY; each also lives in
one C++ source, plan §30.7: production's and 7.9-MR-OPT's in tools/paged_kv_pair.cpp, MA-1's and MA-2's in
tools/paged_kv_pair_ma.cpp):
    production   superslm_pkv_pair row79            PKV_PAIR_PRODUCTION_V2 defs=none   RESERVE, PAIR x 104, IDS
    7.9-MR-OPT   superslm_pkv_pair_mr_opt row79     PKV_PAIR_MR_OPT_V1 defs=none       RESERVE, PAIR x 104, IDS
    MA-1         superslm_pkv_pair_ma ma1           PKV_PAIR_MA1_V1 defs=none          PAIR x 104 only
    MA-2         superslm_pkv_pair_ma ma2           PKV_PAIR_MA2_V1 defs=none          PAIR x 104 only
Production and 7.9-MR-OPT print their RESERVE line and their IDS line, since both belong to the paged arm the guard
grades: a run of either that prints no RESERVE line, or no page ids, is refused (79-7, exit 62; D-SLM8466 (1), T-3210
O-4, D-SLM8494). An MA run that prints either is refused the same way.
The MA executable (plan §29.4, §30.5; the commissioning seat's source, tools/paged_kv_pair_ma.cpp) is
superslm_pkv_pair_ma, linked against the production library superslm only. MA-1 and MA-2 share it (R79A's runs share one
exe_sha256, D-SLM8463); `superslm_pkv_pair_ma identity ma1|ma2` prints the construction's identity line. Its run mode is
`superslm_pkv_pair_ma <ma1|ma2> CAP LAYERS HEADS DIM` plus the five flags the driver appends to every pair command,
`--blob B --model F --process K --pairs 104 --warmup 4` (16 arguments with the program name). It restores the blob into
both of its contiguous-option arms, times W warm-up tokens per arm untimed and then N pairs, and prints, in order: its
identity line first; then one `PAIR <arm_a_ns> <arm_b_ns> <arm_a_token> <arm_b_token>` line per timed pair, arm_a in the
paged column (the statistic is tau_a / tau_b - 1); and no IDS line and no RESERVE line (either from an MA run is
refused, 79-7: both belong to a paged arm). It exits 0. MA-2's offset and the pair-order seed are compile-time
constants in its source, not arguments, and it prints `MA offset_ns=<value> seed=<value>` on stderr. What proves it
linked the production library is its link step, read by tools/check_paged_kv_pair_link.py (plan §30.2).
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import time
import traceback


class Refusal(RuntimeError):
    """A construction or input the driver refuses (cell 7.9's refusal vitality, 79-6 to 79-12). `kind` names the
    per-process refusal (EXIT_CODES' keys) where there is one."""

    def __init__(self, message: str, kind: str | None = None):
        super().__init__(message)
        self.kind = kind


PAIRS_PER_PROCESS = 104
PROCESSES_PER_ROW = 5
WARMUP_TOKENS = 4
BAR = 0.05
LOCK_PATH = Path(r"D:\_scratch\SuperSLM\pkv-c6-timing.lock")

# Rev 16.23 (T-3189 G-4): each refusal under the per-process rule has its own exit code, writes no record for the
# refused process and leaves the row ungraded: 79-6 a blob or manifest mismatch, 79-7 a pair executable that is not
# the construction it was run as (an identity line not the construction's own: a seam library or another
# construction's build; an IDS or RESERVE line from an MA run; a paged run with no page ids or no RESERVE line), 79-8 a
# token
# mismatch, 79-9 a process with other than 104 timed pairs (and a PAIR line that does not parse).
EXIT_CODES = {"blob": 61, "seam": 62, "token": 63, "pairs": 64}
# Not a per-process refusal: the row cannot run at all (the lock held by a live owner, a process that crashed or
# changed boot more than RERUN_LIMIT times, a guard input outside the table, an IDS or RESERVE line that does not parse,
# a driver fault), or an aggregate refused. Bad arguments are refused before anything is touched and write nothing.
# Every other refusal writes no record at a fresh result path and replaces an existing one with a REFUSED record
# (supersede_refused; T-3202 F-12, D-SLM8461).
EXIT_INFRASTRUCTURE = 70
RERUN_LIMIT = 3
PROCESS_CAP_SECONDS = 3600

# What a row's pair command builds (--construction), with the page positions it runs at (plan §23.8 commissioning:
# MA-1 and MA-2 on the option arm at B = 64, 7.9-MR-OPT at B = 16).
CONSTRUCTIONS = {"production": 64, "MA-1": 64, "MA-2": 64, "7.9-MR-OPT": 16}
# D-SLM8463: each aggregate slot's exact set of (construction, row) runs.
AGGREGATE_SLOTS = {
    "R79A": (("MA-1", "R2b"), ("MA-1", "R4a"), ("MA-1", "R4b"), ("MA-2", "R4a")),
    "R79MR": (("7.9-MR-OPT", "R2b"), ("7.9-MR-OPT", "R4a"), ("7.9-MR-OPT", "R4b")),
}

# Cell 7.9's tables (rev 16.21; probes/scatter_guard.py): slot -> {"n": mapped pages after the run, "P": pool pages,
# "m99": the guard bound}, at B = 64.
ROWS = {
    "R2a": {"n": 21, "P": 64, "m99": 3},
    "R2c": {"n": 34, "P": 64, "m99": 4},
    "R2b": {"n": 64, "P": 64, "m99": 6},
    "R4c": {"n": 130, "P": 512, "m99": 4},
    "R4a": {"n": 250, "P": 512, "m99": 5},
    "R4b": {"n": 294, "P": 512, "m99": 5},
}
# slot -> the same at B = 16, for 7.9-MR-OPT (slot R79MR)
MR_OPT_ROWS = {
    "R2b": {"n": 256, "P": 256, "m99": 6},
    "R4a": {"n": 999, "P": 2048, "m99": 5},
    "R4b": {"n": 1175, "P": 2048, "m99": 6},
}

# D-SLM8466 (2): each construction prints its own identity line, first on stdout, so a record shows which construction
# ran without relying on the --construction label. Each string lives here and in one C++ source (plan §30.7):
# production's and 7.9-MR-OPT's in tools/paged_kv_pair.cpp (superslm_pkv_pair, and superslm_pkv_pair_mr_opt built with
# PKV_PAIR_MR_OPT); MA-1's and MA-2's in tools/paged_kv_pair_ma.cpp (superslm_pkv_pair_ma).
IDENTITY = "PKV_PAIR_PRODUCTION_V2 defs=none"
CONSTRUCTION_IDENTITY = {
    "production": IDENTITY,
    "7.9-MR-OPT": "PKV_PAIR_MR_OPT_V1 defs=none",
    "MA-1": "PKV_PAIR_MA1_V1 defs=none",
    "MA-2": "PKV_PAIR_MA2_V1 defs=none",
}
# D-SLM8466 (1): the scatter guard grades the paged arm's page order, so it runs only for the constructions that have a
# paged arm. MA-1 and MA-2 (contiguous-option arms only) are graded by the row rule alone, and an IDS line from one of
# them is refused.
PAGED_ARM = frozenset({"production", "7.9-MR-OPT"})
GUARD_NOT_APPLICABLE = "not applicable (no paged arm)"


# ---- the statistic and the row rule -------------------------------------------------------------------------

def slowdown(paged_ns, contiguous_ns):
    """s_i = tau_paged / tau_contiguous - 1."""
    return float(paged_ns) / float(contiguous_ns) - 1.0


def process_statistic(pairs):
    """pairs: [(paged_ns, contiguous_ns, paged_token, contiguous_token)], exactly 104; the median s_i."""
    pairs = list(pairs)
    if len(pairs) != PAIRS_PER_PROCESS:
        raise Refusal(f"79-9: {len(pairs)} timed pairs, {PAIRS_PER_PROCESS} owed", "pairs")
    values = []
    for i, (paged, contiguous, paged_token, contiguous_token) in enumerate(pairs):
        if paged_token != contiguous_token:
            raise Refusal(f"79-8: token mismatch at pair {i}: paged {paged_token}, contiguous {contiguous_token}",
                          "token")
        if not (math.isfinite(paged) and math.isfinite(contiguous) and paged > 0 and contiguous > 0):
            raise Refusal(f"pair {i}: duration not finite and positive ({paged}, {contiguous})", "pairs")
        values.append(slowdown(paged, contiguous))
    return statistics.median(values)


def row_verdict(process_medians):
    """PASS when the largest of the five process medians is <= 0.05; FAIL when their median is > 0.05;
    NO RESULT otherwise."""
    medians = [float(m) for m in process_medians]
    if len(medians) != PROCESSES_PER_ROW or not all(math.isfinite(m) for m in medians):
        raise Refusal(f"a row is graded on {PROCESSES_PER_ROW} finite process medians, not {medians}")
    if max(medians) <= BAR:
        return "PASS"
    if statistics.median(medians) > BAR:
        return "FAIL"
    return "NO RESULT"


# ---- the scatter guard ----------------------------------------------------------------------------------------

def mode_step(ids):
    """M: the number of neighbour pairs whose step equals the most common step."""
    steps = collections.Counter(ids[j + 1] - ids[j] for j in range(len(ids) - 1))
    return steps.most_common(1)[0][1] if steps else 0


def _table(page_positions):
    if page_positions == 64:
        return ROWS
    if page_positions == 16:
        return MR_OPT_ROWS
    raise Refusal(f"no guard table at B = {page_positions} (64: production, 16: 7.9-MR-OPT)")


def guard_bound(slot, page_positions, n):
    """M99(n, P) for the row at page size 64 (production) or 16 (7.9-MR-OPT). The bound is tabulated at the n each
    row reads after its run; another n has no bound, and is refused rather than guessed."""
    table = _table(page_positions)
    if slot not in table:
        raise Refusal(f"slot {slot} has no guard row at B = {page_positions}")
    row = table[slot]
    if n != row["n"]:
        raise Refusal(f"slot {slot} at B = {page_positions}: the run read n = {n}; the bound is tabulated at "
                      f"n = {row['n']}")
    return row["m99"]


def guard_admits(slot, page_positions, ids):
    """True when the row is graded: M <= M99(n, P) at the n of `ids`."""
    return mode_step(list(ids)) <= guard_bound(slot, page_positions, len(ids))


def id_lists_agree(lists):
    """The five processes' measured-holder id lists must be equal; else the row is NO RESULT."""
    lists = [list(x) for x in lists]
    return bool(lists) and bool(lists[0]) and all(x == lists[0] for x in lists)


# ---- the box -----------------------------------------------------------------------------------------------------

def same_boot(before, after):
    """A process whose LastBootUpTime changed between its start and end is discarded and rerun."""
    return before is not None and before == after


def last_boot():
    """The box's LastBootUpTime, as WMI reports it."""
    if os.name != "nt":
        try:
            return Path("/proc/stat").read_text().split("btime", 1)[1].split()[0]
        except (OSError, IndexError):
            return None
    result = subprocess.run(["powershell", "-NoProfile", "-Command",
                             "(Get-CimInstance Win32_OperatingSystem).LastBootUpTime.ToUniversalTime().ToString('o')"],
                            capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def pid_live(pid):
    if pid <= 0:
        return False
    if os.name == "nt":
        result = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"], capture_output=True,
                                text=True)
        return result.returncode == 0 and re.search(rf',"{pid}"(?:,|$)', result.stdout) is not None
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


# The box timing lock (D-SLM8464). One format, shared with run_paged_kv_pairs.py's PidLock,
# paged_kv_c6_provenance.ps1's Enter-PkvTimingLock and Lock74 in tests/paged-kv/c6_dim7_reset_adopt_timing.cpp: the file
# holds the owner's bare decimal pid and nothing else (the driver's own fields, such as the recovery, go into its row
# record). One protocol in all four programs:
#   - a fresh acquire writes the pid to a temporary file and moves it into place without replacing, so the lock file is
#     never visible empty or half-written;
#   - a file that already exists is recovered only under an exclusive non-blocking OS lock on it; then a file modified
#     before the system's last boot is stale whatever pid it holds, since that pid may now name an unrelated process
#     (D-SLM8466 (3)); otherwise a pid liveness check (any live pid refuses, the caller's own included); then an
#     overwrite with the new owner's pid. An empty file is stale, so a lock left empty by a crash recovers;
#   - content other than ASCII digits naming a pid from 1 to LOCK_PID_MAX, with only space, tab, CR and LF stripped
#     around it (LOCK_TRIM), is torn and refused, in every holder, and so is a file longer than LOCK_MAX_BYTES,
#     whatever its first LOCK_MAX_BYTES hold;
#   - a release unlinks the file when it still holds the releaser's pid, and never truncates it; a file longer than
#     LOCK_MAX_BYTES holds no pid. A reader holding the file open at the release instant leaves the releaser's pid in
#     place, which is stale once the releaser exits.
# The OS lock is one byte far past any content, so a plain reader of the pid is never blocked by it. Lock74 was
# outside this protocol (D-SLM8497) until the 1.12.0 review, with no pre-boot staleness and a POSIX reclaim that
# unlinked the file; it now takes the same steps (tools/test_paged_kv_74_lock.py).
_LOCK_BYTE = 1 << 20
LOCK_ATTEMPTS = 3
# T-3206 N-4: the format's pid is a process id, 1 to 2**31 - 1 (POSIX pid_t, and the Int32 PowerShell's Get-Process
# takes; every Windows pid is in range). The holders read one content the same way: ASCII decimal digits naming a
# pid in that range is a pid (live refuses, dead is stale); empty is stale; anything else, zero and an out-of-range
# number included, is torn and refused by name.
LOCK_PID_MAX = 2**31 - 1
# T-3210 N-1 (D-SLM8494): the only characters stripped around the pid, in every holder. Python's bare strip()
# also strips \x0b, \x0c, \x1c to \x1f and Unicode spaces such as \u3000, which would read "123\x1f" as pid 123.
LOCK_TRIM = " \t\r\n"
# T-3211 O-1 (D-SLM8496): a lock file longer than this is torn, in every holder, on the recovery read and on the
# release check. Each reads at most LOCK_MAX_BYTES + 1 bytes, so the check costs nothing.
LOCK_MAX_BYTES = 4096


def _read_capped(fd):
    """At most LOCK_MAX_BYTES + 1 bytes of `fd` from its start; more than LOCK_MAX_BYTES means the file is too long."""
    os.lseek(fd, 0, os.SEEK_SET)
    data = b""
    while len(data) <= LOCK_MAX_BYTES:
        chunk = os.read(fd, LOCK_MAX_BYTES + 1 - len(data))
        if not chunk:
            break
        data += chunk
    return data


def _os_lock(fd):
    """Take the exclusive, non-blocking OS byte lock on `fd`; False when another process holds it. The OS releases it
    when the holder exits, however it exits."""
    os.lseek(fd, _LOCK_BYTE, os.SEEK_SET)
    try:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.lockf(fd, fcntl.LOCK_EX | fcntl.LOCK_NB, 1, _LOCK_BYTE)
        return True
    except OSError:
        return False


def _os_unlock(fd):
    os.lseek(fd, _LOCK_BYTE, os.SEEK_SET)
    try:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.lockf(fd, fcntl.LOCK_UN, 1, _LOCK_BYTE)
    except OSError:
        pass


def _read_owner(fd):
    """The pid the lock's content names, None when the content is empty (stale). The content is the bare decimal pid
    every leg writes (D-SLM8464): ASCII digits only, naming a pid from 1 to LOCK_PID_MAX. Anything else raises
    ValueError (torn), as in the other two legs (T-3204 N-2; T-3206 N-4, O-3)."""
    data = _read_capped(fd)
    if len(data) > LOCK_MAX_BYTES:
        raise ValueError(f"lock content longer than {LOCK_MAX_BYTES} bytes")
    text = data.decode("utf-8").strip(LOCK_TRIM)
    if not text:
        return None
    value = int(text) if text.isascii() and text.isdigit() else 0
    if not 1 <= value <= LOCK_PID_MAX:
        raise ValueError(f"lock content {text[:60]!r}")
    return value


def boot_epoch():
    """The system's last boot, in seconds since the epoch: now less the uptime (GetTickCount64 on Windows, which counts
    sleep and hibernation; CLOCK_BOOTTIME on Linux). None where neither exists. The same reading in every lock
    holder (D-SLM8466 (3); paged_kv_c6_provenance.ps1 uses [Environment]::TickCount64)."""
    if os.name == "nt":
        import ctypes
        tick = ctypes.windll.kernel32.GetTickCount64
        tick.restype = ctypes.c_uint64
        return time.time() - tick() / 1000.0
    boottime = getattr(time, "CLOCK_BOOTTIME", None)
    return time.time() - time.clock_gettime(boottime) if boottime is not None else None


def _before_boot(fd):
    """D-SLM8466 (3): True when the lock file's modification time is earlier than the last boot. Its pid was recorded
    before the reboot and may now name an unrelated process, so the file is stale whatever pid it holds."""
    boot = boot_epoch()
    return boot is not None and os.fstat(fd).st_mtime < boot


def _publish_lock(lock_path, pid):
    """Create `lock_path` holding `pid`, only when no lock file exists: True when this call created it."""
    temp = lock_path.with_name(f"{lock_path.name}.{pid}.new")
    temp.write_bytes(str(pid).encode("ascii"))
    try:
        if os.name == "nt":
            os.rename(temp, lock_path)  # MoveFileEx without MOVEFILE_REPLACE_EXISTING: fails when the lock exists
        else:
            os.link(temp, lock_path)  # fails when the lock exists
        return True
    except (FileExistsError, PermissionError):
        return False  # an existing (or deleting) lock file: recover it, or retry
    finally:
        temp.unlink(missing_ok=True)


def _recover_lock(lock_path, pid, result_path):
    """Recover an existing lock file for `pid`: under the OS lock, a file modified before the last boot is stale
    whatever it holds (D-SLM8466 (3), checked first); otherwise a live owner refuses, and an empty or dead owner's file
    is stale. A stale file is overwritten with `pid`. Returns {"recovered_from": the stale pid or None, "reason"}, or
    None when the file went away first (retry the fresh acquire)."""
    try:
        fd = os.open(lock_path, os.O_RDWR | getattr(os, "O_BINARY", 0))
    except (FileNotFoundError, PermissionError):
        return None
    try:
        if not _os_lock(fd):
            raise Refusal(f"79-10: the C6 timing lock {lock_path} is being recovered by another runner; second runner "
                          f"(result {result_path}) refused")
        try:
            try:
                if not os.path.samestat(os.fstat(fd), os.stat(lock_path)):
                    return None  # released and replaced after this open
            except FileNotFoundError:
                return None
            if _before_boot(fd):
                try:
                    owner = _read_owner(fd)
                except (OSError, ValueError, UnicodeDecodeError):
                    owner = None
                reason = "before the last boot"
            else:
                try:
                    owner = _read_owner(fd)
                except (OSError, ValueError, UnicodeDecodeError):
                    raise Refusal(f"torn timing lock: {lock_path}") from None
                if owner is not None and pid_live(owner):
                    raise Refusal(f"79-10: the C6 timing lock {lock_path} is held by live pid {owner}; second runner "
                                  f"(result {result_path}) refused")
                reason = "empty" if owner is None else "dead pid"
            os.lseek(fd, 0, os.SEEK_SET)
            os.ftruncate(fd, 0)
            os.write(fd, str(pid).encode("ascii"))
            return {"recovered_from": owner, "reason": reason}
        finally:
            _os_unlock(fd)
    finally:
        os.close(fd)


def _release_lock(lock_path, pid):
    """Unlink the lock when it still holds `pid`; never truncate it. A file another process holds open keeps `pid`."""
    try:
        fd = os.open(lock_path, os.O_RDONLY | getattr(os, "O_BINARY", 0))
        try:
            data = _read_capped(fd)
        finally:
            os.close(fd)
        if len(data) <= LOCK_MAX_BYTES and data.strip(LOCK_TRIM.encode("ascii")) == str(pid).encode("ascii"):
            lock_path.unlink()
    except OSError:
        pass


@contextlib.contextmanager
def timing_lock(result_path, lock_path=LOCK_PATH):
    """Context manager: the box-global C6 timing pid lock, independent of the result path (D-SLM8464's protocol, above).
    A live owner refuses a second runner whatever its result path (79-10); a dead owner's or an empty lock is
    recovered, and the recovery is yielded for the row record ({"recovered_from", "reason", "acquired"}; reason is
    None for a fresh acquire, else "before the last boot", "dead pid" or "empty")."""
    lock_path = Path(lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    pid = os.getpid()
    held = None
    for _ in range(LOCK_ATTEMPTS):
        if _publish_lock(lock_path, pid):
            held = {"recovered_from": None, "reason": None}
            break
        held = _recover_lock(lock_path, pid, result_path)
        if held is not None:
            break
    if held is None:
        raise Refusal(f"79-10: the C6 timing lock {lock_path} could not be acquired in {LOCK_ATTEMPTS} attempts; "
                      f"runner (result {result_path}) refused")
    held["acquired"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    try:
        yield held
    finally:
        _release_lock(lock_path, pid)


# ---- records -----------------------------------------------------------------------------------------------------

def write_record(path, record):
    """Temp-then-rename."""
    path = Path(path)
    temp = path.with_name(path.name + f".{os.getpid()}.partial")
    try:
        temp.write_text(json.dumps(record, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def read_record(path):
    """The record, or None when it is missing or torn (a torn record never supplies a verdict)."""
    try:
        record = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return record if isinstance(record, dict) else None


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# ---- 79-6 and 79-7 ------------------------------------------------------------------------------------------------

def check_row_blob(manifest_path, model_path, restore):
    """79-6: the row's blob checked against its manifest before each process restores it: the manifest chain (each
    piece's start, input digest and own digest), the model binding (the model file's SHA-256) and the stream
    (seed 79, `restore` prompt tokens), and the final blob's digest. Returns (blob path, manifest SHA-256); raises
    Refusal (exit code EXIT_CODES["blob"]) on any mismatch."""
    manifest_path, model_path = Path(manifest_path), Path(model_path)

    def refuse(why):
        raise Refusal(f"79-6: {manifest_path}: {why}", "blob")

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        refuse("missing or torn manifest")
    if not isinstance(manifest, dict) or manifest.get("complete") is not True:
        refuse("the manifest is not complete")
    if not model_path.is_file() or manifest.get("model_sha256") != sha(model_path):
        refuse("the model binding does not match the model file")
    if manifest.get("stream") != {"seed": 79, "restore": int(restore)}:
        refuse(f"the stream is {manifest.get('stream')}, seed 79 and restore {restore} owed")
    pieces = manifest.get("pieces")
    if not isinstance(pieces, list) or not pieces:
        refuse("no pieces")
    at, previous = 0, "-"
    for i, piece in enumerate(pieces):
        path = manifest_path.parent / str(piece.get("file", ""))
        if piece.get("start") != at or piece.get("input_sha256") != previous:
            refuse(f"piece {i}: the chain breaks (start {piece.get('start')}, {at} owed)")
        if not path.is_file() or sha(path) != piece.get("sha256"):
            refuse(f"piece {i} ({path.name}): its digest does not match")
        count = piece.get("count")
        if not isinstance(count, int) or count <= 0:
            refuse(f"piece {i}: count {count}")
        at += count
        previous = piece["sha256"]
    if at != int(restore):
        refuse(f"the chain reaches {at} tokens, {restore} owed")
    blob = manifest_path.parent / str(manifest.get("final_blob", ""))
    if (manifest.get("final_blob") != pieces[-1].get("file") or manifest.get("final_sha256") != pieces[-1].get("sha256")
            or not blob.is_file() or sha(blob) != manifest.get("final_sha256")):
        refuse("the final blob does not match")
    return blob, sha(manifest_path)


def check_build_identity(identity_line, construction="production"):
    """79-7: the pair executable's identity line, checked once per process before timing. Anything but the
    construction's own line (CONSTRUCTION_IDENTITY; production's is "PKV_PAIR_PRODUCTION_V2 defs=none") raises Refusal
    (exit code EXIT_CODES["seam"]): a seam build, or an executable that is not the construction it was run as
    (D-SLM8466 (2))."""
    owed = CONSTRUCTION_IDENTITY[construction]
    if (identity_line or "").strip() != owed:
        raise Refusal(f"79-7: pair executable identity {identity_line!r}, {owed!r} owed for {construction}", "seam")
    return None


# ---- the row ------------------------------------------------------------------------------------------------------

def grade_row(processes):
    """processes: one dict per process, {"status": "ok", "median": float} or {"status": "refused", "kind": k}.
    A row is graded only on five complete processes: {"graded": True, "verdict": row_verdict(...), "record": {...}};
    otherwise {"graded": False, "verdict": None, "record": None, "exit_code": EXIT_CODES[kind of the first refusal]}."""
    for p in processes:
        if p.get("status") != "ok":
            kind = p.get("kind")
            return {"graded": False, "verdict": None, "record": None,
                    "exit_code": EXIT_CODES.get(kind, EXIT_INFRASTRUCTURE)}
    if len(processes) != PROCESSES_PER_ROW:
        return {"graded": False, "verdict": None, "record": None, "exit_code": EXIT_INFRASTRUCTURE}
    medians = [float(p["median"]) for p in processes]
    verdict = row_verdict(medians)
    return {"graded": True, "verdict": verdict,
            "record": {"verdict": verdict, "process_medians": medians, "largest": max(medians),
                       "median_of_medians": statistics.median(medians), "bar": BAR}}


def parse_process_output(text, construction="production"):
    """The pair executable's stdout: the identity line, then PAIR lines, then IDS, and optionally a RESERVE line (the
    paged arm's scatter-reserve construction: `RESERVE seed=<s> singles=<n> source=<header>`). Returns (identity,
    pairs, ids). The RESERVE fields are read by parse_reserve. The identity line must be the construction's own (79-7);
    a construction without a paged arm (MA-1, MA-2) that prints an IDS line (D-SLM8466 (1)) or a RESERVE line (T-3206
    N-2) is refused the same way, and so is a paged construction (production, 7.9-MR-OPT) that prints no page ids or no
    RESERVE line (T-3210 O-4, D-SLM8494)."""
    lines = text.splitlines()
    identity = lines[0] if lines else ""
    check_build_identity(identity, construction)  # 79-7, before any pair is read
    pairs, ids, reserve_seen = [], None, False
    for line in lines[1:]:
        fields = line.split()
        if not fields:
            continue
        if fields[0] == "RESERVE":
            reserve_seen = True
        if fields[0] == "PAIR":
            if len(fields) != 5:
                raise Refusal(f"malformed pair line {line!r}", "pairs")
            try:
                pairs.append((float(fields[1]), float(fields[2]), int(fields[3]), int(fields[4])))
            except ValueError:
                raise Refusal(f"malformed pair line {line!r}", "pairs") from None  # T-3203 N-8: never a raw crash
        elif fields[0] == "RESERVE" and construction not in PAGED_ARM:
            # T-3206 N-2: a reserve is the paged arm's; an MA construction has none to report (79-7, as for IDS)
            raise Refusal(f"79-7: {construction} has no paged arm, and its pair executable printed a RESERVE line "
                          f"{line[:80]!r}: it is not the construction it claims to be", "seam")
        elif fields[0] == "IDS":
            if construction not in PAGED_ARM:
                raise Refusal(f"79-7: {construction} has no paged arm, and its pair executable printed a page-id line "
                              f"{line[:80]!r}: it is not the construction it claims to be", "seam")
            try:
                ids = [int(x) for x in fields[1:]]
            except ValueError:
                raise Refusal(f"INFRASTRUCTURE: malformed IDS line {line[:200]!r}") from None
    if construction in PAGED_ARM and not ids:
        # the mirror of the MA rule (T-3204): a paged construction's guard grades its page ids, so an executable that
        # reports none is not the construction it claims to be (before, the row stood as NO RESULT, "id lists differ")
        raise Refusal(f"79-7: {construction} has a paged arm, and its pair executable printed no page ids", "seam")
    if construction in PAGED_ARM and not reserve_seen:
        # T-3210 O-4 (D-SLM8494): the mirror of the MA rule. A paged construction's ids are drawn over the scattered
        # reserve its RESERVE line names, so an executable that reports none is not the construction it claims to be.
        raise Refusal(f"79-7: {construction} has a paged arm, and its pair executable printed no RESERVE line", "seam")
    return identity, pairs, ids


def parse_reserve(text):
    """The RESERVE line's fields ({"seed", "singles", "source"}), or None when the pair executable reports none
    (T-3202 F-11: every 7.9 verdict is measured over one reserve permutation, and the records name it). A RESERVE line
    whose seed or singles is not an integer raises Refusal (EXIT_INFRASTRUCTURE; T-3203 N-8)."""
    for line in text.splitlines():
        fields = line.split()
        if fields and fields[0] == "RESERVE":
            out = dict(f.split("=", 1) for f in fields[1:] if "=" in f)
            try:
                if "seed" in out:
                    out["seed"] = int(out["seed"])
                if "singles" in out:
                    out["singles"] = int(out["singles"])
            except ValueError:
                raise Refusal(f"INFRASTRUCTURE: malformed RESERVE line {line[:200]!r}") from None
            return out
    return None


def pair_executable(pair_cmd):
    """The pair executable a --pair-cmd runs: its first element, resolved on PATH when it is not a path."""
    first = Path(pair_cmd[0])
    if first.is_file():
        return first.resolve()
    found = shutil.which(pair_cmd[0])
    if not found:
        raise Refusal(f"the pair executable {pair_cmd[0]!r} does not exist")
    return Path(found).resolve()


def main(argv=None):
    """The process loop, cell 7.9's driver, as a command (rev 16.23 G-4; T-3196 F-6), and its aggregate mode
    (D-SLM8463). See the module docstring. Returns the exit code: 0 for a graded row (its verdict PASS, FAIL or NO
    RESULT is in the row record) and for a STANDING aggregate."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--aggregate", action="store_true", help="write R79A's or R79MR's slot record from --runs")
    parser.add_argument("--runs", nargs="+", type=Path, help="--aggregate: the row records the slot covers")
    parser.add_argument("--pair-cmd", help="JSON list: the pair executable's command line")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--restore", type=int)
    parser.add_argument("--slot", required=True)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--lock", type=Path, default=LOCK_PATH)
    parser.add_argument("--page-positions", type=int, default=64, choices=(64, 16))
    parser.add_argument("--construction", default="production", choices=tuple(CONSTRUCTIONS))
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(argv)
    # D-SLM8466 (2): the exact command the row ran, in the row record
    driver_argv = [sys.executable, str(Path(__file__).resolve())] + argv
    if args.aggregate:
        return aggregate_main(args)
    if args.runs is not None:
        print("INFRASTRUCTURE: --runs is for --aggregate", file=sys.stderr)
        return EXIT_INFRASTRUCTURE
    missing = [name for name in ("pair_cmd", "manifest", "model", "restore") if getattr(args, name) is None]
    if missing:
        print(f"INFRASTRUCTURE: a row run needs {', '.join('--' + m.replace('_', '-') for m in missing)}",
              file=sys.stderr)
        return EXIT_INFRASTRUCTURE
    try:
        pair_cmd = json.loads(args.pair_cmd)
        if not isinstance(pair_cmd, list) or not pair_cmd or not all(isinstance(x, str) for x in pair_cmd):
            raise ValueError
    except ValueError:
        print("INFRASTRUCTURE: --pair-cmd is not a JSON list of strings", file=sys.stderr)
        return EXIT_INFRASTRUCTURE
    result = args.result
    if args.slot not in _table(args.page_positions):
        print(f"INFRASTRUCTURE: slot {args.slot} has no row at B = {args.page_positions}", file=sys.stderr)
        return EXIT_INFRASTRUCTURE
    if CONSTRUCTIONS[args.construction] != args.page_positions:
        print(f"INFRASTRUCTURE: construction {args.construction} runs at --page-positions "
              f"{CONSTRUCTIONS[args.construction]}, not {args.page_positions}", file=sys.stderr)
        return EXIT_INFRASTRUCTURE
    stem = result.with_suffix("")
    process_path = lambda k: stem.with_name(f"{stem.name}.p{k}.json")  # noqa: E731
    exe_path, exe_sha = None, None
    try:
        # D-SLM8461: the pair executable's SHA-256, taken by the driver before the first process starts
        exe_path = pair_executable(pair_cmd)
        exe_sha = sha(exe_path)
        with timing_lock(result, args.lock) as lock:
            processes, id_lists, reserves = [], [], []
            for k in range(1, PROCESSES_PER_ROW + 1):
                attempts = 0
                while True:
                    attempts += 1
                    if attempts > RERUN_LIMIT:
                        raise Refusal(f"INFRASTRUCTURE: process {k} failed {RERUN_LIMIT} times")
                    boot_before = last_boot()
                    blob, manifest_sha = check_row_blob(args.manifest, args.model, args.restore)  # 79-6, per process
                    command = pair_cmd + ["--blob", str(blob), "--model", str(args.model), "--process", str(k),
                                          "--pairs", str(PAIRS_PER_PROCESS), "--warmup", str(WARMUP_TOKENS)]
                    try:
                        done = subprocess.run(command, capture_output=True, text=True, timeout=PROCESS_CAP_SECONDS)
                    except subprocess.TimeoutExpired:
                        print(f"process {k}: over {PROCESS_CAP_SECONDS} s; rerun whole", file=sys.stderr)
                        continue
                    # T-3202 F-2: the exit status first. A pair process that crashed (any non-zero exit, before or
                    # after its identity line) is rerun whole; only a clean exit's identity is graded (79-7).
                    if done.returncode != 0:
                        print(f"process {k}: the pair executable exited {done.returncode}; rerun whole\n"
                              f"{done.stderr[-2000:]}", file=sys.stderr)
                        continue
                    identity, pairs, ids = parse_process_output(done.stdout, args.construction)  # 79-7
                    reserve = parse_reserve(done.stdout)
                    median = process_statistic(pairs)  # 79-9, 79-8
                    boot_after = last_boot()
                    if not same_boot(boot_before, boot_after):
                        print(f"process {k}: the boot changed ({boot_before} -> {boot_after}); discarded, rerun",
                              file=sys.stderr)
                        continue
                    break
                record = {"slot": args.slot, "process": k, "complete": True, "median": median,
                          "pairs": [list(p) for p in pairs], "ids": ids, "identity": identity, "boot": boot_before,
                          "blob_sha256": sha(blob), "manifest_sha256": manifest_sha,
                          "model_sha256": sha(args.model), "restore": args.restore, "attempts": attempts,
                          "page_positions": args.page_positions, "construction": args.construction,
                          "argv": command, "exe": str(exe_path), "exe_sha256": exe_sha, "reserve": reserve}
                write_record(process_path(k), record)
                processes.append({"status": "ok", "median": median})
                id_lists.append(ids or [])
                reserves.append(reserve)
            graded = grade_row(processes)
            # D-SLM8461: STANDING exactly when the driver finished the row and wrote a verdict (PASS, FAIL or NO
            # RESULT); the verdict stays its own field. T-3202 F-11: the reserve the ids were read over, with its seed
            # and source, as the pair executable reported it (null where it reports none). D-SLM8464: the lock's
            # acquisition and any recovery are recorded here, never in the lock file.
            row = {"slot": args.slot, "restore": args.restore, "page_positions": args.page_positions,
                   "construction": args.construction, "argv": driver_argv, "identity": identity,
                   "processes": [process_path(k).name for k in range(1, PROCESSES_PER_ROW + 1)],
                   "lock_recovered_from": lock["recovered_from"], "lock_recovered_reason": lock["reason"],
                   "lock_acquired": lock["acquired"],
                   "rule_verdict": graded["verdict"],
                   "status": "STANDING", "exe": str(exe_path), "exe_sha256": exe_sha,
                   "reserve": reserves[0] if all(r == reserves[0] for r in reserves) else reserves}
            row.update(graded["record"] or {})
            if args.construction not in PAGED_ARM:
                # D-SLM8466 (1): no paged arm, no page order to grade; the row rule alone decides
                row.update(guard=GUARD_NOT_APPLICABLE)
            elif not id_lists_agree(id_lists):
                row.update(verdict="NO RESULT", reason="the five measured-holder id lists differ (construction defect)")
            else:
                ids = id_lists[0]
                m99 = guard_bound(args.slot, args.page_positions, len(ids))
                m = mode_step(ids)
                row.update(guard={"n": len(ids), "M": m, "M99": m99, "admits": m <= m99})
                if m > m99:
                    row.update(verdict="NO RESULT", reason="the scatter guard rejects the measured holder's ids")
            write_record(result, row)
            print(f"{args.slot}: {row['verdict']} (process medians {graded['record']['process_medians']})")
            return 0
    except Refusal as refusal:
        print(f"REFUSED: {refusal}", file=sys.stderr)
        code = EXIT_CODES.get(refusal.kind, EXIT_INFRASTRUCTURE)
        supersede_refused(result, args.slot, code, str(refusal), exe_path, exe_sha, args.construction,
                          args.page_positions)
        return code
    except Exception as fault:  # T-3203 N-8: a driver fault refuses through the driver's own code, never a raw crash
        traceback.print_exc()
        reason = f"INFRASTRUCTURE: driver fault {type(fault).__name__}: {fault}"
        print(f"REFUSED: {reason}", file=sys.stderr)
        supersede_refused(result, args.slot, EXIT_INFRASTRUCTURE, reason, exe_path, exe_sha, args.construction,
                          args.page_positions)
        return EXIT_INFRASTRUCTURE


def supersede_refused(result, slot, code, reason, exe_path, exe_sha, construction=None, page_positions=None):
    """T-3202 F-12 with D-SLM8461: a refused run never leaves an earlier run's row record standing at its path. Where a
    record exists at `result`, it is replaced by a REFUSED record carrying the exit code and no verdict; where none
    exists, none is written (a refused row has no row record, cell 7.9's per-process rule)."""
    if not Path(result).exists():
        return
    write_record(result, {"slot": slot, "status": "REFUSED", "exit_code": code, "reason": reason, "verdict": None,
                          "construction": construction, "page_positions": page_positions,
                          "exe": str(exe_path) if exe_path else None, "exe_sha256": exe_sha,
                          "superseded": "the previous row record at this path"})


# ---- the aggregate slot record (D-SLM8463) ------------------------------------------------------------------------

def aggregate_runs(slot, run_paths):
    """R79A's or R79MR's slot record from its runs' row records. Raises Refusal when a run record is missing or torn,
    when the runs are not exactly the slot's set of (construction, row), when a run's page positions are not its
    construction's, or when the runs' exe_sha256 differ (or one is missing). Otherwise returns the record: `status`
    STANDING only when every run is STANDING, REFUSED otherwise."""
    runs = []
    for path in run_paths:
        record = read_record(path)
        if not isinstance(record, dict):
            raise Refusal(f"aggregate {slot}: run record {path} is missing or torn")
        construction, row, status = record.get("construction"), record.get("slot"), record.get("status")
        if status not in ("STANDING", "REFUSED"):
            raise Refusal(f"aggregate {slot}: run record {path} has status {status!r}")
        if construction in CONSTRUCTIONS and record.get("page_positions") != CONSTRUCTIONS[construction]:
            raise Refusal(f"aggregate {slot}: run record {path} ({construction}) at page positions "
                          f"{record.get('page_positions')!r}, {CONSTRUCTIONS[construction]} owed")
        runs.append({"row": row, "construction": construction, "status": status, "verdict": record.get("verdict"),
                     "exe_sha256": record.get("exe_sha256"), "path": str(path)})
    have = sorted((r["construction"] or "", r["row"] or "") for r in runs)
    owed = sorted(AGGREGATE_SLOTS[slot])
    if have != owed:
        raise Refusal(f"aggregate {slot}: the runs are {have}, exactly {owed} owed")
    hashes = {r["exe_sha256"] for r in runs}
    if len(hashes) != 1 or None in hashes:
        raise Refusal(f"aggregate {slot}: the runs' exe_sha256 differ ({sorted(map(str, hashes))})")
    status = "STANDING" if all(r["status"] == "STANDING" for r in runs) else "REFUSED"
    return {"slot": slot, "aggregate": True, "runs": runs, "status": status, "exe_sha256": hashes.pop()}


def aggregate_main(args):
    """--aggregate: write the slot record; 0 when it is STANDING, EXIT_INFRASTRUCTURE when it is REFUSED or refused."""
    if args.slot not in AGGREGATE_SLOTS:
        print(f"INFRASTRUCTURE: --aggregate takes slot {' or '.join(AGGREGATE_SLOTS)}, not {args.slot}",
              file=sys.stderr)
        return EXIT_INFRASTRUCTURE
    if not args.runs:
        print("INFRASTRUCTURE: --aggregate needs --runs", file=sys.stderr)
        return EXIT_INFRASTRUCTURE
    try:
        record = aggregate_runs(args.slot, args.runs)
    except Refusal as refusal:
        print(f"REFUSED: {refusal}", file=sys.stderr)
        supersede_refused(args.result, args.slot, EXIT_INFRASTRUCTURE, str(refusal), None, None)
        return EXIT_INFRASTRUCTURE
    write_record(args.result, record)
    readings = ", ".join("{construction} {row} {status} {verdict}".format(**r) for r in record["runs"])
    print(f"{args.slot}: {record['status']} ({readings})")
    return 0 if record["status"] == "STANDING" else EXIT_INFRASTRUCTURE


if __name__ == "__main__":
    sys.exit(main())
