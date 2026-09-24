TE-425 instrument commissioning constructions (StandardsDocument.md Sec5.4; plan te421-slm172-host-oom.md Sec3.5).
Registered through Claude/Tooling/instrument-commission.ps1 in the records tree. The test author built the
instruments; every construction below is taken from another seat's census, never authored here:
  site sweep (te425_cells.cpp):
    must-accept  the planner's TE-421 census, the 144 operator-new sites v1.7.1 already gets right
                 (own SSLM_GPU_ALLOCATION_FAILED, next encode clean): k 1-88, the six between each
                 recording-window block, and the two at the read (Claude/Vitruvius/te421-probe/run-encode-*.txt).
    must-reject  the planner's TE-421 census, the first recording-window block (k 89-136, all 48 in its
                 480 class: own SSLM_GPU_ALLOCATION_FAILED, the next submission SSLM_DEVICE_LOST); AND the
                 adversary's TE-419 leg B construction, every D3D12 allocation of an encode (here the
                 two-sub-chunk, 5-token encode) returning SSLM_DEVICE_LOST at v1.7.1.
  census check (tests/ci/check_gpu_status_site_census.py):
    must-accept  the committed site list against the v1.7.1 source.
    must-reject  the list with TE-422 S-2's three sites, TE-423 F-5's two exclusion lists and TE-424 P-1's
                 null-token return deleted.
  entry census (te425_cells.cpp r15zero):
    must-accept  the planner's disposition-3 entry points (Sec3.5 R15 item 3) at v1.7.1.
    must-reject  the planner's disposition-2 entry points (Sec3.5 R15 item 2), which allocate host memory.
A construction that crashes must never read as a rejection: each script exits 1 only when the instrument
itself returned its RED verdict (exit 1), and 0 on any other exit.
The paths below are this machine's: the v1.7.1 seam build of the suite (D:\_te425\bin171), the v1.7.1
source (D:\_te425\slm171), the t2791-hash Qwen3 artifact.
