F3 certification on the RX 7900 XTX (SuperSLM branch claude/f3-f4-cells-1p16: the precached-prefixes cells 4.4, 4.6,
6.5 and 10.4 for engine 1.16)

Plug the drive into the work machine, double-click run.cmd, and leave it. It needs nothing installed, no internet and
no admin rights: the executables, shaders, runtime DLLs, fixtures, prompts and models are all on the drive. If the
machine crashes or the window is closed, run run.cmd again: finished steps are skipped. When it prints KIT RUN
FINISHED, bring the drive back to the dev box.

It picks the RX 7900 XTX by itself (vendor 1002 with at least 4 GB; the Ryzen's integrated Radeon, about 0.5 GB, is
skipped) and writes results\<computer name>\ on this drive. The D3D12 debug layer is off for every step. Each step
stops itself at 15 minutes, and the kit rests 5 minutes between steps (and a full 5 minutes after a crash).

Before the three t- steps, close browsers, video players, calls (Teams, Zoom, Slack, Discord) and games. The kit waits
up to 10 minutes for them to close; a speed reading taken while any of them runs is marked busy and thrown away.

  f2set       4.4 here: the six F2 cells, pcg_cell_schema (E0 artifacts), pcg_cell_9_10 (this card's v1.13.1
              captures) and pcg_cell_6_5 on pkv_def's shapes                                     about 3 min
  real15      1.5B: pcg_cell_4_6 (a) CPU vs this GPU, (b) vs the RTX 2080 SUPER's blob, (c) the 2080 SUPER's
              blob restored here through the test seam; then pcg_cell_6_5 on the artifact      about 5 min
  real05      0.5B: the same                                                                   about 3 min
  r10x        10.1 and 10.2 on this GPU: capture, then restore in a second process             about 4 min
  t-acc-15    10.4 speed bar, 1.5B, the must-accept (this build)                                about 3 min
  t-rej-15    10.4 speed bar, 1.5B, the must-reject (the restore-prefills mutant build)        about 6 min
  t-grade-15  10.4 speed bar, 1.5B, the readings to grade                                       about 3 min
  summary.txt one line per step; blobs46\ this card's 4.6 saves (the dev box's cross step reads them)

Expected time: about 27 minutes of steps and 30 minutes of rests, so about one hour. The 4.6 and 10.4 verdicts are
decided on the dev box (tools\precache_gpu\grade_f3.py), not here: a 4.6 difference is a finding, not a broken kit.
