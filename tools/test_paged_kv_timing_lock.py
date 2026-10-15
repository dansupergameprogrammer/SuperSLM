"""Cells for the box timing lock's one format and protocol across its three script programs (T-3203 N-2, D-SLM8464;
SuperSLM 371dc34; the test author's record T-3189 §14). The fourth holder, Lock74 in
tests/paged-kv/c6_dim7_reset_adopt_timing.cpp, has its own cells in tools/test_paged_kv_74_lock.py. Every holder and
every contender is its own process, so each program meets the others' real pids:

  driver   tools/run_paged_kv_rows.py's timing_lock (the 7.9 driver)
  pidlock  tools/run_paged_kv_pairs.py's PidLock (the 7.9b and legacy pair legs)
  ps1      tools/paged_kv_c6_provenance.ps1's Enter-PkvTimingLock / Exit-PkvTimingLock (7.4 and lifecycle), under
           Windows PowerShell 5.1 (`powershell`)
  ps1-core the same script under PowerShell 7 (`pwsh`). The edition is a parameter of the command, so every cell runs
           the PowerShell leg in both (T-3212 C-4): its boot reading takes a different branch in each

  live     a live holder of any program is refused by every program, and the file holds the holder's bare pid
  reader   a release while another process holds the file open leaves the releaser's pid (never an empty file); once
           both have exited, every program recovers it, and its own release removes the file
  killed   a holder killed hard is recovered by every program
  empty    an empty lock file is stale: every program acquires it
  oslock   another process holding the OS byte (offset 1 MiB) over a dead pid: every program refuses
  race     four concurrent contenders of one program over a dead owner's file: exactly one holds
  ownpid   the driver refuses a lock that names its own live pid (it is not stale)
  release  a release unlinks only a file that still holds the releaser's pid
  publish  (the two Python programs, by audit hook) a fresh acquire never opens the lock path itself: the path's first
           and only event is the move (rename, or link) of the temporary file holding the pid into place, so the lock
           is never visible empty. The PowerShell acquire is not observable this way (record T-3189 §14).
  boot     (D-SLM8466 (3)) a lock file modified before the last boot is stale in every holder, whatever it holds (a live
           pid, torn content); the same live pid written after the boot refuses; the driver records the reason
  torn     (T-3204 N-2) a JSON body, or a dead pid with a sign or a digit separator, is torn in every holder; (T-3206
           N-4, O-3) so is a number outside 1 to 2**31 - 1 (99999999999, 2147483648, 0, 00) and non-ASCII digits
           (fullwidth); a dead 2147483647, the largest pid, is recovered by every holder
  strip    (T-3210 N-1, D-SLM8494) exactly space, tab, CR and LF are stripped around the pid: a dead pid's digits beside
           \x1f, U+3000, \x1c, \x0b, \x0c or U+00A0 are torn in every holder; '<pid>\r\n' and ' <pid>\t' still read
           as the pid (recovered when dead). A release removes '<own pid>\r\n' and leaves '<own pid>\x0b'
  sharing  (T-3204 O-3) a holder of the file without write sharing: every holder refuses, PowerShell by name
  length   (T-3211 O-1, D-SLM8496 (2)) a lock file longer than 4096 bytes is torn in every holder, on the recovery read
           and on the release check, whatever its first 4096 bytes hold: a dead pid then 4200 spaces then 'x'; a dead
           pid padded with spaces to 4097 bytes; a dead pid placed after 4097 spaces. A file of exactly 4096 bytes is
           read as before (a dead pid padded to 4096 is recovered; '<own pid>' padded to 4096 is released), and a
           4097-byte file modified before the last boot is stale (D-SLM8466 (3): whatever it holds)
  edition  (T-3212 C-4, T-3206 O-1) each PowerShell edition runs as itself and takes its own branch of
           Get-PkvBootTimeUtc: Windows PowerShell 5.1 (Desktop) has no [Environment]::TickCount64 and reads the CIM
           boot time; PowerShell 7 (Core) reads TickCount64. Each edition's boot reading is within BOOT_TOLERANCE_S of
           this file's own reading (now less GetTickCount64)

PKV_ROWS_DRIVER, PKV_PAIRS_RUNNER and PKV_PROVENANCE_PS1 name the programs under test (default: this checkout's); the
test author runs the remedy reverted through them.
"""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest

TOOLS = Path(__file__).resolve().parent
DRIVER = Path(os.environ.get("PKV_ROWS_DRIVER") or TOOLS / "run_paged_kv_rows.py")
PAIRS = Path(os.environ.get("PKV_PAIRS_RUNNER") or TOOLS / "run_paged_kv_pairs.py")
PS1 = Path(os.environ.get("PKV_PROVENANCE_PS1") or TOOLS / "paged_kv_c6_provenance.ps1")
# The PowerShell leg's programs and the executable that runs each (T-3212 C-4): R5's command names no edition
PS_EDITIONS = {"ps1": "powershell", "ps1-core": "pwsh"}
PROGRAMS = ("driver", "pidlock", "ps1", "ps1-core")

PY_AGENT = textwrap.dedent(r'''
    import importlib.util, sys
    from pathlib import Path
    program, path, lock, action = sys.argv[1], sys.argv[2], Path(sys.argv[3]), sys.argv[4]
    spec = importlib.util.spec_from_file_location("leg", path)
    leg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(leg)
    events = []
    if action == "audit":
        target = str(lock)
        def hook(event, args):
            # open: (path, mode, flags); os.rename / os.link / os.replace: (src, dst, ...)
            if event == "open" and str(args[0]) == target:
                events.append(["open", str(args[0]), str(args[1]), str(args[2])])
            elif event in ("os.rename", "os.link", "os.replace") and target in (str(args[0]), str(args[1])):
                events.append([event, str(args[0]), str(args[1])])
        sys.addaudithook(hook)
    try:
        if program == "driver":
            ctx = leg.timing_lock(lock.with_name("result.json"), lock_path=lock)
        else:
            ctx = leg.PidLock(lock)
        ctx.__enter__()
    except leg.Refusal as refusal:
        print("REFUSED", refusal, flush=True)
        sys.exit(3)
    if action == "audit":
        import json
        print("HELD " + json.dumps(events), flush=True)
    else:
        print("HELD", flush=True)
    if action == "hold":
        sys.stdin.readline()
    ctx.__exit__(None, None, None)
    print("RELEASED", flush=True)
''')

PS_AGENT = textwrap.dedent(r'''
    param([string]$Script, [string]$Lock, [string]$Action)
    $ErrorActionPreference = 'Stop'
    . $Script
    try { $null = Enter-PkvTimingLock -Path $Lock } catch { [Console]::Out.WriteLine("REFUSED $($_.Exception.Message)"); [Console]::Out.Flush(); exit 3 }
    [Console]::Out.WriteLine("HELD"); [Console]::Out.Flush()
    if ($Action -eq 'hold') { $null = [Console]::In.ReadLine() }
    Exit-PkvTimingLock -Path $Lock
    [Console]::Out.WriteLine("RELEASED"); [Console]::Out.Flush()
''')

# The edition, whether [Environment]::TickCount64 exists (the branch Get-PkvBootTimeUtc takes), and its boot reading in
# milliseconds since the Unix epoch, UTC
PS_BOOT = textwrap.dedent(r'''
    param([string]$Script)
    $ErrorActionPreference = 'Stop'
    . $Script
    $boot = Get-PkvBootTimeUtc
    $ticks = $null
    try { $ticks = [Environment]::TickCount64 } catch { $ticks = $null }
    $ms = ($boot.ToUniversalTime() - [DateTime]::SpecifyKind([DateTime]'1970-01-01', 'Utc')).TotalMilliseconds
    [Console]::Out.WriteLine("$($PSVersionTable.PSEdition) $($PSVersionTable.PSVersion.Major) $([int]($null -ne $ticks)) $([long]$ms) $($boot.Kind)")
''')


class TimingLock(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="t3189-lock-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.lock = self.root / "pkv-c6-timing.lock"
        (self.root / "agent.py").write_text(PY_AGENT)
        (self.root / "agent.ps1").write_text(PS_AGENT)
        (self.root / "boot.ps1").write_text(PS_BOOT)
        self.procs = []
        self.addCleanup(self.cleanup)

    def cleanup(self):
        for p in self.procs:
            if p.poll() is None:
                p.kill()
            p.wait()
            for s in (p.stdin, p.stdout):
                if s:
                    s.close()

    def command(self, program, action):
        if program in PS_EDITIONS:
            return [PS_EDITIONS[program], "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(self.root / "agent.ps1"),
                    "-Script", str(PS1), "-Lock", str(self.lock), "-Action", action]
        return [sys.executable, "-I", str(self.root / "agent.py"), program, str(DRIVER if program == "driver" else PAIRS),
                str(self.lock), action]

    def start(self, program, action="hold"):
        p = subprocess.Popen(self.command(program, action), stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        self.procs.append(p)
        return p

    def first_line(self, p):
        return (p.stdout.readline() or "").strip()

    def hold(self, program):
        """A holder process: returns it once it reports HELD."""
        p = self.start(program)
        line = self.first_line(p)
        self.assertEqual(line, "HELD", f"{program} could not take a fresh lock: {line}")
        return p

    def release(self, p):
        p.stdin.write("release\n")
        p.stdin.flush()
        self.assertEqual(self.first_line(p), "RELEASED")
        p.wait(timeout=60)

    def attempt(self, program, action="once"):
        """One acquire-and-release attempt in its own process: 'HELD' or 'REFUSED ...'."""
        done = subprocess.run(self.command(program, action), capture_output=True, text=True, timeout=120)
        return (done.stdout.strip().splitlines() or [""])[0]

    def dead_pid(self):
        p = subprocess.Popen([sys.executable, "-I", "-c", "pass"])
        p.wait()
        return p.pid

    def content(self):
        return self.lock.read_bytes().decode("ascii", "replace") if self.lock.exists() else None

    # live: every program refuses every program's live holder; the file holds the bare pid
    def test_live_holder_is_refused_by_every_program(self):
        for holder in PROGRAMS:
            h = self.hold(holder)
            self.assertEqual(self.content(), str(h.pid), f"{holder}'s lock file is not its bare pid")
            for contender in PROGRAMS:
                with self.subTest(holder=holder, contender=contender):
                    self.assertTrue(self.attempt(contender).startswith("REFUSED"), f"{contender} took {holder}'s lock")
            self.release(h)
            self.assertFalse(self.lock.exists(), f"{holder}'s release left the file")

    # reader: a release with the file held open by a reader leaves the releaser's pid; once both exit, every program
    # recovers it and its own release removes the file
    def test_release_with_a_reader_leaves_the_pid_then_recovers(self):
        for holder in PROGRAMS:
            for contender in PROGRAMS:
                with self.subTest(holder=holder, contender=contender):
                    h = self.hold(holder)
                    reader = subprocess.Popen([sys.executable, "-I", "-c",
                                               "import sys,time; f=open(sys.argv[1],'rb'); print('open', flush=True); "
                                               "sys.stdin.readline()", str(self.lock)],
                                              stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
                    self.procs.append(reader)
                    self.assertEqual(self.first_line(reader), "open")
                    self.release(h)
                    self.assertEqual(self.content(), str(h.pid), "the release left no pid, or an empty file")
                    reader.stdin.write("done\n")
                    reader.stdin.flush()
                    reader.wait(timeout=30)
                    self.assertEqual(self.attempt(contender), "HELD", f"{contender} did not recover {holder}'s pid")
                    self.assertFalse(self.lock.exists(), f"{contender}'s release did not remove the file")

    # killed: a holder killed hard is recovered by every program
    def test_killed_holder_is_recovered(self):
        for holder in PROGRAMS:
            for contender in PROGRAMS:
                with self.subTest(holder=holder, contender=contender):
                    h = self.hold(holder)
                    h.kill()
                    h.wait()
                    self.assertEqual(self.attempt(contender), "HELD")
                    self.assertFalse(self.lock.exists())

    # empty: an empty lock file is stale
    def test_empty_file_is_stale(self):
        for contender in PROGRAMS:
            with self.subTest(contender=contender):
                self.lock.write_bytes(b"")
                self.assertEqual(self.attempt(contender), "HELD")
                self.assertFalse(self.lock.exists())

    # oslock: another process holding the OS byte over a dead pid makes every program refuse
    def test_os_byte_held_over_a_dead_pid_refuses(self):
        self.lock.write_text(str(self.dead_pid()))
        byte = subprocess.Popen([sys.executable, "-I", "-c", textwrap.dedent('''
            import msvcrt, os, sys
            fd = os.open(sys.argv[1], os.O_RDWR | os.O_BINARY)
            os.lseek(fd, 1 << 20, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            print("byte", flush=True)
            sys.stdin.readline()
            ''') if os.name == "nt" else textwrap.dedent('''
            import fcntl, os, sys
            fd = os.open(sys.argv[1], os.O_RDWR)
            fcntl.lockf(fd, fcntl.LOCK_EX | fcntl.LOCK_NB, 1, 1 << 20)
            print("byte", flush=True)
            sys.stdin.readline()
            '''), str(self.lock)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        self.procs.append(byte)
        self.assertEqual(self.first_line(byte), "byte")
        for contender in PROGRAMS:
            with self.subTest(contender=contender):
                self.assertTrue(self.attempt(contender).startswith("REFUSED"))

    # race: four concurrent contenders of one program over a dead owner's file: exactly one holds
    def test_concurrent_recoverers_yield_one_holder(self):
        for program in PROGRAMS:
            with self.subTest(program=program):
                self.lock.write_text(str(self.dead_pid()))
                contenders = [self.start(program) for _ in range(4)]
                lines = [self.first_line(p) for p in contenders]
                holders = [p for p, line in zip(contenders, lines) if line == "HELD"]
                self.assertEqual(len(holders), 1, f"{program}: {lines}")
                for p in holders:
                    self.release(p)
                for p in contenders:
                    p.wait(timeout=60)

    # ownpid: the driver does not treat its own live pid as recoverable
    def test_driver_refuses_a_lock_naming_its_own_pid(self):
        spec = importlib.util.spec_from_file_location("rows_driver", DRIVER)
        rows = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rows)
        self.lock.write_text(str(os.getpid()))
        with self.assertRaises(rows.Refusal):
            with rows.timing_lock(self.root / "r.json", lock_path=self.lock):
                pass
        self.assertEqual(self.content(), str(os.getpid()))

    # release: a release unlinks only a file that still holds the releaser's pid
    def test_release_unlinks_only_its_own_pid(self):
        for holder in PROGRAMS:
            with self.subTest(holder=holder):
                h = self.hold(holder)
                other = str(self.dead_pid())
                self.lock.write_text(other)  # the file now names another pid
                self.release(h)
                self.assertEqual(self.content(), other, f"{holder}'s release removed or changed another pid's lock")
                self.lock.unlink()

    # publish: a fresh acquire moves the pid into place; it never opens the lock path to create or write it
    def test_fresh_acquire_moves_the_pid_into_place(self):
        import json
        for program in ("driver", "pidlock"):
            with self.subTest(program=program):
                line = self.attempt(program, "audit")
                self.assertTrue(line.startswith("HELD "), line)
                events = json.loads(line[5:])
                self.assertTrue(events, f"{program}: no event touched the lock path")
                first = events[0]
                self.assertIn(first[0], ("os.rename", "os.link", "os.replace"), f"{program} opened the lock: {events}")
                self.assertEqual(first[2], str(self.lock), f"{program}: the move's target is not the lock: {first}")
                self.assertNotEqual(first[0], "os.replace", f"{program}: the move replaces an existing lock")
                # after the move, only the release's read of the pid touches the path
                self.assertFalse([e for e in events[1:] if e[0] != "open" or e[2] not in ("r", "rb")],
                                 f"{program} wrote the lock after moving it into place: {events}")
                self.assertFalse(self.lock.exists())

    # ---- T-3204 (second confirmation review) remedies, D-SLM8466 (3) and N-2/O-3, SuperSLM 4156b30 --------------------
    def last_boot(self):
        """The last boot in epoch seconds, this cell's own reading: now less the uptime."""
        if os.name == "nt":
            import ctypes
            tick = ctypes.windll.kernel32.GetTickCount64
            tick.restype = ctypes.c_uint64
            return time.time() - tick() / 1000.0
        return time.time() - time.clock_gettime(time.CLOCK_BOOTTIME)

    def live_pid(self):
        p = subprocess.Popen([sys.executable, "-I", "-c", "import sys; sys.stdin.readline()"], stdin=subprocess.PIPE,
                             text=True)
        self.procs.append(p)
        return p.pid

    def write_lock(self, text, before_boot):
        self.lock.write_bytes(text.encode("utf-8"))
        if before_boot:
            when = self.last_boot() - 3600
            os.utime(self.lock, (when, when))

    # boot: a lock file modified before the last boot is stale whatever it holds, a live pid or torn content included;
    # the same live pid written after the boot still refuses
    def test_lock_modified_before_the_last_boot_is_stale(self):
        live = self.live_pid()
        for contender in PROGRAMS:
            for name, text, before, want in (("live pid, after the boot", str(live), False, "REFUSED"),
                                             ("live pid, before the boot", str(live), True, "HELD"),
                                             ("torn, before the boot", '{"pid": 1', True, "HELD")):
                with self.subTest(contender=contender, case=name):
                    self.write_lock(text, before)
                    line = self.attempt(contender)
                    self.assertTrue(line.startswith(want), f"{contender}, {name}: {line}")
                    if want == "HELD":
                        self.assertFalse(self.lock.exists())
                    self.lock.unlink(missing_ok=True)

    # boot, the driver's record: the recovery's reason is "before the last boot"
    def test_driver_records_a_pre_boot_recovery(self):
        spec = importlib.util.spec_from_file_location("rows_driver_boot", DRIVER)
        rows = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rows)
        self.write_lock(str(self.live_pid()), True)
        with rows.timing_lock(self.root / "r.json", lock_path=self.lock) as held:
            self.assertEqual(held.get("reason"), "before the last boot")

    # torn: the content is the bare decimal pid and nothing else. A JSON body, or a dead pid with a sign or a digit
    # separator, is torn in every holder (an int() or TryParse reading would recover the dead pid)
    def test_json_and_signed_pids_are_torn(self):
        dead = self.dead_pid()
        digits = str(dead)
        for contender in PROGRAMS:
            for name, text in (("json", '{"pid": %d}' % dead), ("plus sign", "+" + digits),
                               ("separator", digits[:1] + "_" + digits[1:]), ("minus sign", "-" + digits),
                               ("above Int32", "99999999999"), ("2**31", "2147483648"), ("zero", "0"),
                               ("zeros", "00"), ("fullwidth digits", "\uff11\uff12\uff13")):
                with self.subTest(contender=contender, content=name):
                    self.write_lock(text, False)
                    line = self.attempt(contender)
                    self.assertTrue(line.startswith("REFUSED"), f"{contender} took a lock holding {text!r}: {line}")
                    self.assertIn("torn", line)
                    self.assertEqual(self.lock.read_bytes(), text.encode("utf-8"), f"{contender} changed a torn lock")
                    self.lock.unlink()

    # range: 2147483647 (2**31 - 1) is a pid; dead, it is stale, recovered by every holder, and its release removes it
    def test_largest_pid_dead_is_recovered(self):
        for contender in PROGRAMS:
            with self.subTest(contender=contender):
                self.write_lock("2147483647", False)
                self.assertEqual(self.attempt(contender), "HELD")
                self.assertFalse(self.lock.exists())

    # strip: only space, tab, CR and LF are stripped (T-3210 N-1). The digits are a dead pid's, so a holder that strips
    # more reads a dead pid and recovers the file (HELD), where it must refuse it as torn. PowerShell's bare Trim() read
    # \x0b and \x0c as the pid (the builder's §22.1); Python's strip() read all six
    def test_only_space_tab_cr_lf_are_stripped(self):
        digits = str(self.dead_pid())
        for contender in PROGRAMS:
            for name, text in (("unit separator", digits + "\x1f"), ("ideographic space", digits + "\u3000"),
                               ("file separator before", "\x1c" + digits), ("vertical tab", digits + "\x0b"),
                               ("form feed", digits + "\x0c"), ("no-break space", digits + "\u00a0")):
                with self.subTest(contender=contender, around=name):
                    self.write_lock(text, False)
                    line = self.attempt(contender)
                    self.assertTrue(line.startswith("REFUSED"), f"{contender} took a lock holding {text!r}: {line}")
                    self.assertIn("torn", line)
                    self.assertEqual(self.lock.read_bytes(), text.encode("utf-8"), f"{contender} changed a torn lock")
                    self.lock.unlink()
            for name, text in (("CR LF after", digits + "\r\n"), ("space before, tab after", " " + digits + "\t")):
                with self.subTest(contender=contender, around=name):
                    self.write_lock(text, False)
                    self.assertEqual(self.attempt(contender), "HELD", f"{contender} did not read {text!r} as a dead pid")
                    self.assertFalse(self.lock.exists())

    # strip, on release: the own-pid check strips the same set. '<own pid>\r\n' is the holder's and is removed;
    # '<own pid>\x0b' is not, and stays
    def test_release_strips_only_space_tab_cr_lf(self):
        for holder in PROGRAMS:
            for suffix, removed in (("\r\n", True), ("\x0b", False), ("\x0c", False)):
                with self.subTest(holder=holder, suffix=repr(suffix)):
                    h = self.hold(holder)
                    text = str(h.pid) + suffix
                    self.lock.write_bytes(text.encode("ascii"))
                    self.release(h)
                    if removed:
                        self.assertFalse(self.lock.exists(), f"{holder} kept its own pid with {suffix!r}")
                    else:
                        self.assertTrue(self.lock.exists(), f"{holder}'s release removed a lock holding {text!r}")
                        self.assertEqual(self.lock.read_bytes(), text.encode("ascii"),
                                         f"{holder}'s release removed a lock holding {text!r}")
                        self.lock.unlink()

    # sharing: another process holds the lock file open without write sharing. Every holder refuses; the PowerShell
    # holder names the sharing conflict (never a raw .NET error)
    def test_sharing_conflict_refuses_and_powershell_names_it(self):
        self.write_lock(str(self.dead_pid()), False)
        script = self.root / "share.ps1"
        script.write_text("param([string]$Lock)\n"
                          "$s = [IO.File]::Open($Lock, 'Open', 'Read', 'Read')\n"
                          "[Console]::Out.WriteLine('open'); [Console]::Out.Flush()\n"
                          "$null = [Console]::In.ReadLine(); $s.Dispose()\n")
        holder = subprocess.Popen(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script),
                                   "-Lock", str(self.lock)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        self.procs.append(holder)
        self.assertEqual(self.first_line(holder), "open")
        for contender in PROGRAMS:
            with self.subTest(contender=contender):
                line = self.attempt(contender)
                self.assertTrue(line.startswith("REFUSED"), f"{contender}: {line}")
                if contender in PS_EDITIONS:
                    self.assertIn("sharing conflict", line)
                    self.assertIn("3 attempts", line)

    # ---- T-3211 O-1 (D-SLM8496 (2)): a lock file longer than 4096 bytes is torn in every holder --------------------
    # The digits are a dead pid's, so a holder that reads a window of the file and finds the pid there (or finds only
    # stripped characters, read as empty) recovers it (HELD), where it must refuse it as torn.
    LOCK_SPAN = 4096

    def test_lock_longer_than_4096_bytes_is_torn(self):
        digits = str(self.dead_pid())
        span = self.LOCK_SPAN
        torn = (("dead pid, 4200 spaces, x (T-3211 O-1)", digits + " " * 4200 + "x"),
                ("dead pid padded with spaces to 4097 bytes", digits + " " * (span + 1 - len(digits))),
                ("dead pid after 4097 spaces", " " * (span + 1) + digits))
        for contender in PROGRAMS:
            for name, text in torn:
                with self.subTest(contender=contender, content=name):
                    self.assertGreater(len(text.encode("ascii")), span)
                    self.write_lock(text, False)
                    line = self.attempt(contender)
                    self.assertTrue(line.startswith("REFUSED"),
                                    f"{contender} took a {len(text)}-byte lock ({name}): {line}")
                    self.assertIn("torn", line)
                    self.assertEqual(self.lock.read_bytes(), text.encode("ascii"), f"{contender} changed a torn lock")
                    self.lock.unlink()
            with self.subTest(contender=contender, content="dead pid padded with spaces to exactly 4096 bytes"):
                text = digits + " " * (span - len(digits))
                self.assertEqual(len(text.encode("ascii")), span)
                self.write_lock(text, False)
                self.assertEqual(self.attempt(contender), "HELD", f"{contender} did not recover a 4096-byte dead pid")
                self.assertFalse(self.lock.exists())
            with self.subTest(contender=contender, content="4097 bytes, modified before the last boot"):
                self.write_lock(digits + " " * (span + 1 - len(digits)), True)
                line = self.attempt(contender)
                self.assertEqual(line, "HELD", f"{contender} did not take a pre-boot 4097-byte lock: {line}")
                self.assertFalse(self.lock.exists())

    # length, on release: '<own pid>' padded with spaces past 4096 bytes is torn, not the holder's, and stays;
    # padded to exactly 4096 bytes it is the holder's and is removed
    def test_release_leaves_a_lock_longer_than_4096_bytes(self):
        span = self.LOCK_SPAN
        for holder in PROGRAMS:
            for size, removed in ((span, True), (span + 1, False)):
                with self.subTest(holder=holder, size=size):
                    h = self.hold(holder)
                    text = str(h.pid).ljust(size)
                    self.assertEqual(len(text), size)
                    self.lock.write_bytes(text.encode("ascii"))
                    self.release(h)
                    if removed:
                        self.assertFalse(self.lock.exists(), f"{holder} kept its own pid padded to {size} bytes")
                    else:
                        self.assertTrue(self.lock.exists(),
                                        f"{holder}'s release removed a {size}-byte lock holding its pid")
                        self.assertEqual(self.lock.read_bytes(), text.encode("ascii"),
                                         f"{holder}'s release changed a {size}-byte lock")
                        self.lock.unlink()

    # ---- T-3212 C-4 (T-3206 O-1): the PowerShell leg's boot reading in each edition --------------------------------
    # The tolerance. Measured 2026-10-11 on the bench (Windows 10, Windows PowerShell 5.1, PowerShell 7.6.6): PowerShell
    # 7's TickCount64 reading 0.001 to 0.004 s from this file's; 5.1's CIM boot time 2.40 s from it (T-3206 measured
    # 0.72 s on its box). The CIM time is the boot instant as recorded; "now less the uptime" moves with every clock
    # correction since the boot, so their gap grows with uptime. 120 s holds weeks of such drift, and is 30 times
    # smaller than the 3,600 s by which the pre-boot cells place a file before the boot. A wrong unit, sign, time zone
    # or property in either branch is hours off
    BOOT_TOLERANCE_S = 120

    def test_powershell_boot_reading_in_each_edition(self):
        want = {"ps1": ("Desktop", 5, 0), "ps1-core": ("Core", None, 1)}
        for program, exe in PS_EDITIONS.items():
            with self.subTest(program=program, exe=exe):
                done = subprocess.run([exe, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                                       str(self.root / "boot.ps1"), "-Script", str(PS1)],
                                      capture_output=True, text=True, timeout=120)
                ours = self.last_boot()
                self.assertEqual(done.returncode, 0, f"{exe}: {done.stdout}{done.stderr}")
                fields = done.stdout.split()
                self.assertEqual(len(fields), 5, f"{exe}: {done.stdout!r}")
                edition, major, has_ticks, ms, kind = fields[0], int(fields[1]), int(fields[2]), int(fields[3]), fields[4]
                edition_want, major_want, ticks_want = want[program]
                self.assertEqual(edition, edition_want, f"{exe} is not the {edition_want} edition")
                if major_want is not None:
                    self.assertEqual(major, major_want, f"{exe} is PowerShell {major}")
                else:
                    self.assertGreaterEqual(major, 7, f"{exe} is PowerShell {major}")
                self.assertEqual(has_ticks, ticks_want,
                                 f"{exe}: [Environment]::TickCount64 {'exists' if has_ticks else 'is absent'}, so "
                                 f"Get-PkvBootTimeUtc takes the other branch")
                self.assertEqual(kind, "Utc", f"{exe}: Get-PkvBootTimeUtc returned a {kind} time")
                gap = ms / 1000.0 - ours
                self.assertLessEqual(abs(gap), self.BOOT_TOLERANCE_S,
                                     f"{exe}'s boot reading is {gap:+.3f} s from now less GetTickCount64")


if __name__ == "__main__":
    unittest.main()
