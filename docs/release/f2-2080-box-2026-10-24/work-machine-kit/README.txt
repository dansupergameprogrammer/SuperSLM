F2 GPU twins on the RX 7900 XTX (SuperSLM main 5599f7e4; the GPU prefix save and restore, for engine 1.16.0)

Plug the drive into the work machine, double-click run.cmd, and leave it. It needs nothing installed, no internet
and no admin rights: the executables, shaders, runtime DLLs, fixtures and models are all on the drive. If the machine
crashes or the window is closed, run run.cmd again: finished steps are skipped. When it prints KIT RUN FINISHED,
bring the drive back to the dev box.

It picks the RX 7900 XTX by itself (vendor 1002 with at least 4 GB; the Ryzen's integrated GPU is skipped) and writes
results\<computer name>\ on this drive:

  f2      the six F2 cells with the D3D12 debug layer (format, pool, busy, faults, budget snapshot, restore-derived)
  schema  pcg_cell_schema with the E0 artifacts
  cross   the RTX 2080 SUPER's saved prefix blob restored here: it must be refused SSLM_RESTORE_PRODUCER_MISMATCH;
          and this card's own blob saved (blob-1002_<device>.spx1), for the reverse check on the dev box
  g5, g4  pkvg_g5 and pkvg_g4, their own G5 and G4 cells, with the debug layer
  g19     cell 1.9 alone, without the debug layer (it took about 90 minutes under it on the 2080 SUPER)
  summary.txt   one line per step

The debug layer needs Windows' "Graphics Tools" optional feature. If it is not installed the kit says so and runs the
cells without it; the summary records which.

Expected time: about 15 to 30 minutes, depending mostly on 1.9.
