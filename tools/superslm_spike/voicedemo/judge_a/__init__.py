"""P2: Judge-A validation harness (`RuntimeLoRA_VoiceDemo_Plan.md` Sec.8 P2). Judge A is a bounded
yes/no entailment extractor (canonical design Sec.5): "does this reaction convey required_info[i]?
does it carry intended_attitude?" -- never open-ended "rate the fidelity."

See `backend.py`'s module docstring for the standing execution blocker: this harness is fully built
and self-tested against `MockJudgeBackend`, but has not been run against a real pinned judge model,
because no judge model has been selected or provisioned with API access. That is filed as a blocker
in the T-1980 build log, not improvised around.
"""
