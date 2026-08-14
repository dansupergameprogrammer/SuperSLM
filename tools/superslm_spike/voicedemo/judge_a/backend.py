"""Judge-A backend interface, plus the two implementations this build actually has.

**Standing execution blocker (filed in the T-1980 build log, `Claude/Brunel/` in the Wizard repo --
not improvised around per this dispatch's own BLOCKERS instruction):** the plan requires a "pinned
judge, temp 0, disclosed... a different model than the base" (canonical design Sec.5). No such model
has been selected or provisioned with API access as of this build. `HostedApiJudgeBackend` below is
the shape a real backend takes, gated on an environment variable that is not set in this environment
-- it raises `JudgeBackendUnavailable` rather than silently falling back to an unpinned or
undisclosed model. `validate.py`'s harness runs end-to-end and is fully exercised against
`MockJudgeBackend`, which proves the precision/recall/CI/escalation machinery is correct; it has not
produced real numbers against a real judge, because there is no real judge wired up yet.
"""

from __future__ import annotations

import abc
import os
import random


class JudgeBackendUnavailable(RuntimeError):
    """Raised when a backend that needs external provisioning (an API key, a hosted model, a local
    weights file) is asked to run without it. Never caught and silently substituted -- the harness
    surfaces this as a blocker, per StandardsDocument.md Sec.5.3 (do not present unverified work as
    verified) and this dispatch's own instruction to file, not improvise around, blockers."""


class JudgeABackend(abc.ABC):
    """One bounded yes/no call per question. `model_id` is recorded alongside every verdict so the
    corpus/validation record discloses which judge produced it (plan Sec.6.1's disclosure
    requirement, extended to Judge A)."""

    model_id: str

    @abc.abstractmethod
    def answer_info(self, reaction_text: str, required_info_item: str) -> bool:
        """True if the judge says the reaction conveys the given required_info item."""

    @abc.abstractmethod
    def answer_attitude(self, reaction_text: str, intended_attitude: str) -> bool:
        """True if the judge says the reaction carries the given intended_attitude."""


class MockJudgeBackend(JudgeABackend):
    """A deterministic stand-in judge for testing the P2 harness's own logic (precision/recall
    computation, CI math, iterate-and-escalate control flow) independent of any real model. Two
    modes:

    - `error_rate=0.0` (default): a "perfect" judge that reads this corpus's own ground-truth
      construction directly (it knows which validation reactions are planted negatives because the
      harness's self-test controls the whole pipeline) -- used to prove the precision/recall
      computation reports 1.0/1.0 when the judge is in fact perfect.
    - `error_rate>0.0`: flips the judge's answer for a deterministic, seeded fraction of calls --
      used to prove the harness correctly measures and reports an imperfect judge's precision/recall,
      and correctly drives the iterate-then-escalate path when the measured lower-CI-bound does not
      clear the plan's 0.85 bar (Sec.8 P2).

    This backend is not, and is not claimed to be, a real judge model. It never produces a number
    that this package's own output represents as Judge A's real measured precision/recall.
    """

    model_id = "mock-judge-v1 (test double, not a real judge model)"

    def __init__(self, error_rate: float = 0.0, seed: int = 0) -> None:
        if not 0.0 <= error_rate <= 1.0:
            raise ValueError(f"error_rate must be in [0, 1], got {error_rate}")
        self.error_rate = error_rate
        self._seed = seed
        # A seeded `random.Random` gives a well-distributed, reproducible-per-seed stream. An
        # earlier version of this method built its own linear hash of (seed, call_count); that hash
        # advanced too slowly relative to its own range to decorrelate consecutive calls, so a
        # fixed seed produced either near-zero or near-total flips across an entire run instead of
        # the intended ~error_rate fraction (caught by test_judge_validate_harness.py's escalation
        # cell measuring exactly 0.0 or 1.0 observed precision/recall instead of something near
        # 1-error_rate; StandardsDocument.md Sec.5.4 -- exactness verified by execution, not by
        # construction). `random.Random` is the standard fix: verified empirically at construction
        # time in this module's own tests to land within a few points of `error_rate` over the
        # corpus's ~240 calls.
        self._rng = random.Random(seed)

    def _maybe_flip(self, truth: bool) -> bool:
        if self.error_rate <= 0.0:
            return truth
        flip = self._rng.random() < self.error_rate
        return (not truth) if flip else truth

    def answer_info(self, reaction_text: str, required_info_item: str) -> bool:
        # The mock "reads" ground truth via a marker the harness's caller supplies through
        # answer_info_with_truth; plain answer_info without known truth is not meaningful for a
        # mock and is not used by validate.py (which always calls with truth available). Kept for
        # interface completeness only.
        raise NotImplementedError("MockJudgeBackend requires ground truth; use answer_info_with_truth via validate.py")

    def answer_attitude(self, reaction_text: str, intended_attitude: str) -> bool:
        raise NotImplementedError("MockJudgeBackend requires ground truth; use answer_attitude_with_truth via validate.py")

    def answer_info_with_truth(self, ground_truth: bool) -> bool:
        return self._maybe_flip(ground_truth)

    def answer_attitude_with_truth(self, ground_truth: bool) -> bool:
        return self._maybe_flip(ground_truth)


class HostedApiJudgeBackend(JudgeABackend):
    """The real-backend shape: calls a pinned, temperature-0, disclosed hosted model. Gated on
    `VOICEDEMO_JUDGE_A_API_KEY` (and `VOICEDEMO_JUDGE_A_MODEL_ID` to record which pinned version) --
    neither is set in this build's environment, so constructing this backend raises
    `JudgeBackendUnavailable` immediately rather than making an unauthorized or unbudgeted call.
    Which provider/model to pin, and the API spend to provision it, is the T-1980 blocker filed in
    the build log."""

    def __init__(self) -> None:
        api_key = os.environ.get("VOICEDEMO_JUDGE_A_API_KEY")
        model_id = os.environ.get("VOICEDEMO_JUDGE_A_MODEL_ID")
        if not api_key or not model_id:
            raise JudgeBackendUnavailable(
                "HostedApiJudgeBackend requires VOICEDEMO_JUDGE_A_API_KEY and "
                "VOICEDEMO_JUDGE_A_MODEL_ID to be set (a pinned, disclosed judge model different "
                "from the SLM base model, per canonical design Sec.5). Neither is provisioned in "
                "this environment -- filed as a blocker in the T-1980 build log, not improvised "
                "around with an unpinned or undisclosed model."
            )
        self.model_id = model_id
        self._api_key = api_key

    def answer_info(self, reaction_text: str, required_info_item: str) -> bool:
        raise NotImplementedError(
            "HostedApiJudgeBackend's call implementation is not written -- it cannot be constructed "
            "in this environment (see __init__), so there is nothing to test it against yet."
        )

    def answer_attitude(self, reaction_text: str, intended_attitude: str) -> bool:
        raise NotImplementedError(
            "HostedApiJudgeBackend's call implementation is not written -- it cannot be constructed "
            "in this environment (see __init__), so there is nothing to test it against yet."
        )
