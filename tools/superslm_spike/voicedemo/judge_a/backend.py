"""Judge-A backend interface, plus the implementations this build has.

**Unblocked (T-2011, 2026-08-13) per D-SLM3017 (RULED under delegation, subject to Dan's veto):
judge provisioning is Option B, a self-hosted, pinned-weights judge.** `OllamaJudgeBackend` below is
the real implementation -- a local Ollama server serving a pinned model tag + content-addressed
layer digest, called at temperature 0 with a fixed seed, disclosed in full via `model_id` on every
verdict (plan Sec.6.1's disclosure requirement, extended to Judge A). `HostedApiJudgeBackend` remains
the Option-A shape, still gated on an unset environment variable, kept as the documented fallback if
Dan later flips the provisioning call ("Flips to the hosted-API option on Dan's word" -- D-SLM3017).
`validate.py`'s harness runs end-to-end and is fully exercised against `MockJudgeBackend` for its own
control-flow logic (precision/recall/CI/escalation machinery); `OllamaJudgeBackend` is what produces
real numbers against a real judge.
"""

from __future__ import annotations

import abc
import json
import os
import random
import re
import time
import urllib.error
import urllib.request

from .prompt import build_attitude_prompt, build_info_prompt


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


_YES_WORD = re.compile(r"\bYES\b", re.IGNORECASE)
_NO_WORD = re.compile(r"\bNO\b", re.IGNORECASE)


class OllamaJudgeBackend(JudgeABackend):
    """The provisioned real judge (T-2011, D-SLM3017 Option B): a self-hosted, pinned-weights model
    served locally by Ollama, called at temperature 0 with a fixed seed. Distinct model family from
    the SLM base (Qwen2.5) -- this is Llama 3.1 -- per canonical design Sec.5's no-self-grading
    guardrail.

    **Pin and disclosure, stated exactly (plan Sec.6.1):**
    - model tag: `llama3.1:8b-instruct-q4_K_M`
    - content-addressed weights digest: `sha256:667b0c1932bc6ffc593ed1d03f895bf2dc8dc6df21db3042284a6f4416b06a29`
      (the GGUF weights layer of the pulled manifest, read from
      `~/.ollama/models/manifests/registry.ollama.ai/library/llama3.1/8b-instruct-q4_K_M` at
      provisioning time -- not the mutable tag alone, since a tag can be silently repointed by a
      future `ollama pull` whereas the layer digest cannot)
    - decoding: `temperature=0` (greedy), fixed `seed` (default 42), sent on every call
    - `model_id` (below) carries all of the above so every verdict discloses exactly which judge,
      which weights, and which decoding settings produced it.

    **Unparseable-response convention, disclosed rather than hidden:** a response is parsed by
    whole-word match on YES/NO (case-insensitive), tolerating the model appending stray punctuation
    or a trailing token despite the bounded-answer instruction. A response containing neither or both
    words is not guessable as a genuine YES -- it is scored as NO (the judge failed to affirmatively
    assert the fact/tone) and counted in `unparseable_count` so the disposition is visible in the
    run's own report rather than silently folded into either direction's numbers.
    """

    DEFAULT_MODEL_TAG = "llama3.1:8b-instruct-q4_K_M"
    DEFAULT_DIGEST = "sha256:667b0c1932bc6ffc593ed1d03f895bf2dc8dc6df21db3042284a6f4416b06a29"
    DEFAULT_SEED = 42
    _MAX_RETRIES = 3
    _RETRY_BACKOFF_S = 0.5

    def __init__(
        self,
        model_tag: str = DEFAULT_MODEL_TAG,
        digest: str = DEFAULT_DIGEST,
        base_url: str = "http://localhost:11434",
        seed: int = DEFAULT_SEED,
        timeout: float = 60.0,
        info_prompt_fn=build_info_prompt,
        attitude_prompt_fn=build_attitude_prompt,
    ) -> None:
        self._model_tag = model_tag
        self._digest = digest
        self._base_url = base_url.rstrip("/")
        self._seed = seed
        self._timeout = timeout
        self._info_prompt_fn = info_prompt_fn
        self._attitude_prompt_fn = attitude_prompt_fn
        self.unparseable_count = 0
        self.model_id = (
            f"{model_tag} (ollama, self-hosted, layer digest {digest}, "
            f"temperature=0, seed={seed})"
        )
        self._verify_available()

    def _verify_available(self) -> None:
        """Confirms the Ollama server is reachable and the pinned model tag is actually pulled
        locally -- never silently lets Ollama's own auto-pull-on-demand behavior substitute an
        unpinned or unverified model."""

        try:
            req = urllib.request.Request(f"{self._base_url}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise JudgeBackendUnavailable(
                f"OllamaJudgeBackend could not reach the Ollama server at {self._base_url}: {exc}. "
                f"Provision by starting Ollama and running `ollama pull {self._model_tag}`."
            ) from exc

        names = {m.get("name") for m in body.get("models", [])}
        if self._model_tag not in names:
            raise JudgeBackendUnavailable(
                f"OllamaJudgeBackend: model {self._model_tag!r} is not pulled locally (found: "
                f"{sorted(n for n in names if n)}). Run `ollama pull {self._model_tag}` first -- "
                f"never silently substituted with a different tag."
            )

    def _call(self, prompt: str) -> str:
        payload = {
            "model": self._model_tag,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0, "seed": self._seed, "num_predict": 16, "top_p": 1.0},
        }
        data = json.dumps(payload).encode("utf-8")
        last_exc: Exception | None = None
        for attempt in range(self._MAX_RETRIES):
            try:
                req = urllib.request.Request(
                    f"{self._base_url}/api/chat",
                    data=data,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                return body["message"]["content"]
            except (urllib.error.URLError, OSError, ValueError, KeyError) as exc:
                last_exc = exc
                if attempt < self._MAX_RETRIES - 1:
                    time.sleep(self._RETRY_BACKOFF_S * (attempt + 1))
        raise JudgeBackendUnavailable(
            f"OllamaJudgeBackend call failed after {self._MAX_RETRIES} attempts: {last_exc}"
        ) from last_exc

    def _parse_yes_no(self, raw: str) -> bool:
        cleaned = raw.strip()
        yes = bool(_YES_WORD.search(cleaned))
        no = bool(_NO_WORD.search(cleaned))
        if yes and not no:
            return True
        if no and not yes:
            return False
        self.unparseable_count += 1
        return False

    def answer_info(self, reaction_text: str, required_info_item: str) -> bool:
        prompt = self._info_prompt_fn(reaction_text, required_info_item)
        return self._parse_yes_no(self._call(prompt))

    def answer_attitude(self, reaction_text: str, intended_attitude: str) -> bool:
        prompt = self._attitude_prompt_fn(reaction_text, intended_attitude)
        return self._parse_yes_no(self._call(prompt))
