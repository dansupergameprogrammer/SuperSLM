T-1872 PORTABLE PROBE -- what this is and how to run it
=========================================================

WHAT THIS CHECKS

This checks one question: can the 7900 XTX in this machine run our model at
all, and how fast? It does NOT do the real measurement campaign -- it is a
few-minutes check to find out whether that's even worth building.

Four things get checked, in order:
  1. Does PyTorch see the GPU as the right chip (gfx1100)?
  2. Does bf16 math work on it?
  3. Can it run our actual model on one real sentence and get a sane answer?
  4. How many seconds per document does it take -- so we can compare its
     speed against the RTX 2080 Super this project already measures on?

If a step can't run, it says so and moves on (or stops, if there's no point
continuing). Nothing hangs forever -- every step gives up after a while if
something's stuck, instead of leaving you staring at a frozen window.

HOW TO RUN IT

1. Plug in the drive.
2. Open this folder.
3. Double-click "run_probe.bat".
4. It will unpack itself onto this machine's own local disk first (into a
   temporary folder under your Temp folder), THEN run there -- running the
   model straight off a USB drive would be slow, so it copies itself
   somewhere fast first. This needs about 9 GB of free space on your main
   drive, temporarily. It checks that space is available before doing
   anything, and says so plainly if it isn't.
5. A black window opens and prints progress. Let it run -- it can take up to
   roughly half an hour in the worst case (every check timing out), but that
   is a pessimistic ceiling, not a typical run; most runs finish in a few
   minutes.
6. When it's done, it says "Done" and waits for you to press a key.
7. Look at "RESULTS.txt" in this same folder ON THE DRIVE -- that's the
   answer, in plain English, no digging required.

WHAT IT DOES AND DOESN'T DO TO YOUR MACHINE

- It copies itself (about 9 GB) into a temporary folder under your Temp
  folder to run from there, and DELETES that temporary folder automatically
  when it finishes -- whether it succeeded or failed. Nothing is left behind
  on this machine's own disk either way.
- The only thing that's written back permanently is on the DRIVE itself:
  RESULTS.txt and a results\ folder next to it, both plain text/JSON.
- It does not install anything system-wide, does not touch your PATH or the
  Windows registry, does not update any driver.
- It needs no internet connection -- everything it needs is already on the
  drive, packed into the single T1872_Probe.tar file next to this README.
- The only thing it needs from your machine that isn't on the drive: your
  existing AMD graphics driver has to be new enough for ROCm 7.2.1 (the
  AMD docs this bundle was built from say driver version 26.2.2 or newer),
  and Windows' own tar.exe (built in since Windows 10, nothing to install).
  If step 1 fails, an out-of-date driver is one of the first things to check
  -- RESULTS.txt will say what actually happened either way.

IF SOMETHING FAILS

RESULTS.txt tells you, in plain language, which of the four checks failed
and why. You don't need to fix it yourself -- just send RESULTS.txt (and,
if asked, the results\full_log.txt file) back and it'll get sorted out from
there. If it fails before RESULTS.txt is even written, the black window's
own on-screen text is the next best thing to send back.
