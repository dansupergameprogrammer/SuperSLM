# Long-leg drive kit

Runs SuperSLM 1.14.0's long CPU legs on a spare Windows PC instead of the work machine. It is built for
the i5-14400F (32 GB, Windows 11 Home) but runs on any 64-bit Windows 10 or 11 PC.

## What it runs

Each row below is one *piece* per toolchain. The toolchains are MSVC 14.34, clang-cl 15 and MSVC with
AddressSanitizer, in that order.

| Cell | Toolchains | Hours per piece (one core) |
|---|---|---|
| 4.1/E2:synthetic-shapes (the 1.5B grid) | MSVC | about 6.6 |
| 4.2/E3 | all three | about 7 |
| 4.7/E2:synthetic1p5 (unpinned) | all three | about 2.3 |
| 9.8/E2:synthetic1p5 (unpinned) | all three | about 2.0 |
| 4.7/E3:synthetic1p5 (unpinned) | all three | about 1.0 |
| 9.8/E3:synthetic1p5 (1.5B) | all three | about 0.75 |

The AddressSanitizer pieces take about 2.5 times as long. In total that is about 65 CPU-hours. With four
pieces at once, that is **about 21 hours** (anywhere from 16 to 28). The MSVC pieces run first and take
about 7 to 8 hours of that.

These are not run here, and the summary lists them with the reason:

- The two TSan pkv_32k cells. They are Linux-only, because there is no ThreadSanitizer on Windows.
- E4's timing and real-artifact cells. They are blocked until the reference re-derivation, the licensed
  references and the timing instrument exist. The real models go on the drive anyway, so the drive is
  ready for them.

## Drive

Use a **32 GB** drive formatted as NTFS or exFAT.

- The inputs take about 5 GB: the fixtures, the E0 artifacts, the two models and a git bundle of the source.
- The builds and checkpoint journals take about 7 GB more. They go to `C:\_scratch\longleg` when C: has
  30 GB free, otherwise onto the drive.

## 1. Fill the drive (home box, once)

Plug the drive in. In PowerShell, run this, using the drive's letter:

    powershell -NoProfile -ExecutionPolicy Bypass -File D:\SuperSLM\tools\longleg_kit\fill_drive.ps1 -Drive F:

It does the following:

- copies the files from `D:\_artifacts\SuperSLM` and checks each one against its recorded SHA-256;
- bundles the branch's full history, so the PC needs no GitHub login;
- copies this kit as committed, and puts `START.cmd` at the root of the drive.

It takes a few minutes. When it is done, it prints the size.

## 2. Run it (spare PC)

Plug the drive in and **double-click `START.cmd`** on the drive.

The first time, it lists what the PC is missing and gives the exact line that installs each one. On a
fresh Windows install, all of these are missing:

- Visual Studio 2022 Build Tools, with MSVC 14.34, the Windows 11 SDK and AddressSanitizer
- LLVM 15 (clang-cl)
- Git
- Python 3
- CMake
- Ninja

Type `Y` to install them all. Visual Studio asks for permission once and takes 10 to 20 minutes. Then
double-click `START.cmd` again.

After that, leave the PC alone. It builds once per toolchain (about 10 minutes each), then runs the
pieces. The window shows each piece as it starts and finishes. The PC stays awake while it runs.

**A crash, a reboot or a closed window is fine.** At the next logon it carries on by itself, from a line
it adds to your Startup folder. You can also double-click `START.cmd` again. Finished pieces are never
rerun, and the two long grids resume at the case they were on. Do not update the drive's kit or the
source in the middle of a run, because the grids' journals are tied to the exact build.

## 3. How to tell it is done

These two things show it is done:

- `F:\longleg\results\ALL-DONE.txt` exists;
- the top of `F:\longleg\results\SUMMARY.md` says **All pieces are done**.

The Startup line removes itself at that point. Bring back `SUMMARY.md`; it is the one file that gets
committed. Keep the rest of the drive until the summary has been read, because the per-piece logs are in
`results\` too.

To check progress at any time, open `SUMMARY.md`, which is rewritten after every piece. A piece marked
**RED** or **CRASHED** is not a kit failure. It is a result, and its first failure lines are in the summary.

## Options

Pass these to `START.cmd` or `longleg.ps1`:

| Option | What it does |
|---|---|
| `-CheckOnly` | Checks the prerequisites and the drive, then stops. |
| `-Toolchains msvc` | Runs only the MSVC pieces (about 7 to 8 hours). |
| `-Parallel 6` | Runs more pieces at once. The default is 4. |
| `-Only msvc__4.2` | Runs only the pieces whose id starts with this. |
| `-RetryCrashed` | Runs again the pieces recorded **CRASHED** (the old record and logs are kept, renamed). |
| `-SelfTest` | Tests the scheduler with stand-in pieces. It builds nothing. |

To run one piece again, delete its `results\<piece>.done` file and start the kit.

After each build the kit starts both programs once (`--list`, which runs no cell). A program Windows cannot
start fails its build in seconds, with the exit code in `results\build-<toolchain>.done`, instead of crashing
every piece. For the AddressSanitizer build, the kit first copies MSVC 14.34's ASan runtime
(`clang_rt.asan_dynamic-x86_64.dll`) beside the programs, because Windows looks in the program's own folder
first. On the first spare-PC run, without this, every ASan piece exited 0xC0000142 (DLL initialization failed)
at once. To re-run those pieces with the fixed kit, once the rest are done: copy the new kit onto the drive and
run `START.cmd -Toolchains msvc-asan -RetryCrashed`.
