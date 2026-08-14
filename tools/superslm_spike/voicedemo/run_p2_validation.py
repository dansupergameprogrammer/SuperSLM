#!/usr/bin/env python3
"""T-2011: run P2's real Judge-A validation against the provisioned self-hosted judge
(`OllamaJudgeBackend`, D-SLM3017 Option B) over the corpus's 120 human-labeled `validation_reactions`.

Run with `python -m voicedemo.run_p2_validation` from `Tools/superslm_spike/` (same convention as
`build_corpus.py`). Requires the corpus already built (`python -m voicedemo.build_corpus`) and an
Ollama server running locally with `llama3.1:8b-instruct-q4_K_M` pulled.

Prints per-call progress (foreground-watched, per this project's no-unwatched-backgrounding
practice) and, at the end, the full precision/recall/CI table plus the plan's PASS/ESCALATE verdict
(plan Sec.8 P2 / Sec.10: precision >= 0.85 and recall >= 0.85, lower Wilson CI bound, on both
`info_present` and `attitude_present`).
"""

from __future__ import annotations

import os
import sys
import time

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_THIS_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

from voicedemo.event_bank import build_entries  # noqa: E402
from voicedemo.judge_a.backend import OllamaJudgeBackend  # noqa: E402
from voicedemo.judge_a.validate import run_validation  # noqa: E402
from voicedemo.validation_reactions import attach_validation_reactions  # noqa: E402

# Iteration 2/3 prompt variants (plan Sec.8 P2 / Risk #12: "iterate the judge prompt (bounded to 3
# iterations)"). Iteration 1 uses judge_a/prompt.py's canonical templates via OllamaJudgeBackend's
# own defaults. These variants are reworded, not just re-emphasized, on the theory that a genuinely
# different phrasing -- not the same instruction repeated -- is what could plausibly move a
# deterministic (temperature-0) judge's answer on a case it got wrong the first time.


def _info_prompt_v2(reaction_text: str, required_info_item: str) -> str:
    return (
        "Read the dialogue line below. Then decide: does it state or clearly imply the fact given? "
        "Reply with a single word, YES or NO, and nothing else.\n\n"
        f"Fact: {required_info_item}\n\n"
        f'Dialogue: "{reaction_text}"\n\n'
        "Single-word answer (YES/NO):"
    )


def _attitude_prompt_v2(reaction_text: str, intended_attitude: str) -> str:
    return (
        "Read the dialogue line below. Then decide: does its tone match the description given? "
        "Reply with a single word, YES or NO, and nothing else.\n\n"
        f"Tone description: {intended_attitude}\n\n"
        f'Dialogue: "{reaction_text}"\n\n'
        "Single-word answer (YES/NO):"
    )


def _info_prompt_v3(reaction_text: str, required_info_item: str) -> str:
    return (
        "Task: fact-checking. Below is one fact and one line of dialogue. If the dialogue "
        "communicates the fact -- even using different words -- the correct answer is YES. If the "
        "dialogue does not communicate the fact, the correct answer is NO. Reply with only YES or "
        "only NO.\n\n"
        f"Fact: {required_info_item}\n"
        f'Dialogue: "{reaction_text}"\n'
        "Answer:"
    )


def _attitude_prompt_v3(reaction_text: str, intended_attitude: str) -> str:
    return (
        "Task: tone-checking. Below is one target tone and one line of dialogue. If the dialogue's "
        "tone matches the target -- even if the wording differs -- the correct answer is YES. If it "
        "does not match, the correct answer is NO. Reply with only YES or only NO.\n\n"
        f"Target tone: {intended_attitude}\n"
        f'Dialogue: "{reaction_text}"\n'
        "Answer:"
    )


_PROMPT_VARIANTS = [
    (None, None),  # iteration 1: OllamaJudgeBackend's own defaults (judge_a/prompt.py canonical)
    (_info_prompt_v2, _attitude_prompt_v2),
    (_info_prompt_v3, _attitude_prompt_v3),
]


class _ProgressOllamaJudgeBackend(OllamaJudgeBackend):
    """Adds foreground progress printing on top of `OllamaJudgeBackend` -- no change to judging
    logic, only visibility into an in-flight run (T-811's no-unwatched-backgrounding practice)."""

    _call_count = 0
    _total_expected = 0  # set by the runner before use

    def answer_info(self, reaction_text: str, required_info_item: str) -> bool:
        result = super().answer_info(reaction_text, required_info_item)
        self._progress_tick()
        return result

    def answer_attitude(self, reaction_text: str, intended_attitude: str) -> bool:
        result = super().answer_attitude(reaction_text, intended_attitude)
        self._progress_tick()
        return result

    def _progress_tick(self) -> None:
        type(self)._call_count += 1
        n, total = type(self)._call_count, type(self)._total_expected
        if n == 1 or n % 20 == 0 or n == total:
            print(f"  ... {n}/{total} judge calls complete", flush=True)


def main() -> int:
    entries = attach_validation_reactions(build_entries())
    n_reactions = sum(len(e.validation_reactions) for e in entries)
    n_calls_per_iteration = n_reactions * 2
    print(f"P2 real validation: {n_reactions} validation reactions, "
          f"{n_calls_per_iteration} judge calls per iteration (info + attitude), up to 3 iterations.")

    backends_by_iteration: dict[int, _ProgressOllamaJudgeBackend] = {}

    def backend_factory(iteration: int) -> _ProgressOllamaJudgeBackend:
        info_fn, attitude_fn = _PROMPT_VARIANTS[iteration - 1]
        kwargs = {}
        if info_fn is not None:
            kwargs["info_prompt_fn"] = info_fn
        if attitude_fn is not None:
            kwargs["attitude_prompt_fn"] = attitude_fn
        print(f"\n=== Iteration {iteration}: provisioning judge "
              f"({'canonical prompt' if info_fn is None else 'reworded prompt v' + str(iteration)}) ===")
        backend = _ProgressOllamaJudgeBackend(**kwargs)
        print(f"Judge disclosed as: {backend.model_id}")
        type(backend)._call_count = 0
        type(backend)._total_expected = n_calls_per_iteration
        backends_by_iteration[iteration] = backend
        return backend

    t0 = time.time()
    outcome = run_validation(entries, backend_factory)
    elapsed = time.time() - t0

    print(f"\n=== Result: {outcome.verdict} ({len(outcome.iterations)} iteration(s), "
          f"{elapsed:.1f}s total) ===")
    for it in outcome.iterations:
        print(f"\n-- Iteration {it.iteration} (model: {it.model_id}) --")
        for field, fr in it.field_results.items():
            print(
                f"  {field}: tp={fr.tp} fp={fr.fp} tn={fr.tn} fn={fr.fn}  "
                f"precision={fr.precision_ci.point_estimate:.4f} "
                f"(95% CI [{fr.precision_ci.lower:.4f}, {fr.precision_ci.upper:.4f}])  "
                f"recall={fr.recall_ci.point_estimate:.4f} "
                f"(95% CI [{fr.recall_ci.lower:.4f}, {fr.recall_ci.upper:.4f}])  "
                f"{'PASS' if fr.passes else 'FAIL'} (bar: lower CI >= 0.85)"
            )
        backend = backends_by_iteration.get(it.iteration)
        if backend is not None:
            print(f"  unparseable judge responses this iteration (scored NO, disclosed not hidden): "
                  f"{backend.unparseable_count}")

    if outcome.verdict == "ESCALATE":
        print(f"\n{outcome.escalation_message}")

    return 0 if outcome.verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
