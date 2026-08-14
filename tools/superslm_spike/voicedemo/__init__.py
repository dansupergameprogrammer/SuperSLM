"""Runtime-LoRA voice-cast demonstration (T-1980) -- corpus + Judge-A validation tooling.

Builds P1 (corpus authoring) and P2 (Judge-A validation) of
`Claude/Plans/RuntimeLoRA_VoiceDemo_Plan.md` (Wizard repo) against the canonical design at
`Claude/Mendeleev/runtime-lora-demo-voice-cast-test-design-2026-07-20.md`. This package does not
restate either document; it implements what they specify. See the plan's Sec.4-Sec.9 for the
schema, guardrails, and calibration numbers this code is built to.

Scope: P1 (corpus) and P2 (Judge-A validation harness). P3 (adapter training) and later phases are
out of scope for this package as built -- see `judge_a/backend.py` for the P2 execution blocker
(no pinned judge model / API access provisioned yet).
"""
