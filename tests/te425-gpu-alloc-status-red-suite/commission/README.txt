TE-425 instrument commissioning constructions (StandardsDocument.md Sec5.4; plan te421-slm172-host-oom.md Sec3.5).
Registered through Claude/Tooling/instrument-commission.ps1 in the records tree. The test author built the
instruments; every construction below is taken from another seat's census or the builder's own engine, never
authored here:
  site sweep (te425_cells.cpp):
    must-accept  the planner's TE-421 census, the 144 operator-new sites v1.7.1 already gets right
                 (own SSLM_GPU_ALLOCATION_FAILED, next encode clean): k 1-88, the six between each
                 recording-window block, and the two at the read (Claude/Vitruvius/te421-probe/run-encode-*.txt).
    must-reject  the planner's TE-421 census, the first recording-window block (k 89-136, all 48 in its
                 480 class: own SSLM_GPU_ALLOCATION_FAILED, the next submission SSLM_DEVICE_LOST); AND the
                 adversary's TE-419 leg B construction, every D3D12 allocation of an encode (here the
                 two-sub-chunk, 5-token encode) returning SSLM_DEVICE_LOST at v1.7.1.
  census check (tests/ci/check_gpu_status_site_census.py):
    must-accept  the committed site list against the builder's current tip (042bd66, the TE-432 fix round).
                 First commissioned against v1.7.1, then at dad862e, then at 042bd66; each time the test author
                 re-listed the census (the graded seat never edits its grader).
    must-reject  the list with TE-422 S-2's three sites, TE-423 F-5's two exclusion lists and TE-424 P-1's
                 null-token return deleted (12 rows at dad862e and at 042bd66, 10 at v1.7.1), selected by the
                 census test's POPULATION predicates.
  entry census (te425_cells.cpp r15zero):
    must-accept  the planner's disposition-3 entry points (Sec3.5 R15 item 3) at v1.7.1.
    must-reject  the planner's disposition-2 entry points (Sec3.5 R15 item 2), which allocate host memory.
  tail pin (te433_tail_pin.cpp, TE-433):
    must-accept  the builder's TE-432 tip 042bd66, both tails' catch (...) present: every leg (decode-step tail
                 T13; both sub-chunk tails T15 of a 5-token prompt; a foreign exception and a length_error).
    must-reject  042bd66 with one tail's catch (...) deleted by reversing the builder's own hunk of 6383435 --
                 the tail exactly as it stood at 7dda90a, where the builder's diagnostic build located the
                 defect -- each driven at its own tail, both kinds (6 legs).
A construction that crashes must never read as a rejection: each script exits 1 only when the instrument
itself returned its RED verdict (exit 1), and 0 on any other exit.
The paths below are this machine's: the v1.7.1 seam build of the suite (D:\_te425\bin171), the v1.7.1
source (D:\_te425\slm171), the 042bd66 source for the census check (D:\_te433\src-042bd66, git archive),
the tail pin's three builds (D:\_te433\final-tip, final-mutT13, final-mutT15: build_engine.bat then
build_red_suite.bat over D:\_te433\src-042bd66, src-mutT13, src-mutT15), the t2791-hash Qwen3 artifact.
  foreign-exception sweep (te436_foreign_sweep.cpp, TE-436):
    must-accept  at c3b5412: the code reviewer's TE-431 prompt-route legs (the final sub-chunk's two
                 finish sites of a 5-token prompt, the last two sites of the prefill phase) and the
                 builder's TE-432 tails, which wait a submission out (every outstanding site of the decode
                 step and of both prompt sub-chunks) -- fx_sweep_must_accept.bat <bin-c3b5412> <qwen3>.
    must-reject  the code reviewer's TE-431 S-1 legs (the decode step's two finish-phase sites at c3b5412,
                 the first of them at v1.7.1: the finish lets a foreign exception escape and the handle's
                 reset is refused), and the builder's TE-432 tail finding at v1.7.1 (the decode step's first
                 outstanding site, the in-flight token's allocation, gated) --
                 fx_sweep_must_reject.bat <bin-c3b5412> <bin-v1.7.1> <qwen3>. Sites are named by --select,
                 not by number; both scripts take their build directories as arguments, and the registry
                 entry names them.