"""The P2 harness itself (`RuntimeLoRA_VoiceDemo_Plan.md` Sec.8 P2 / Sec.10): run Judge A against the
corpus's human-labeled `validation_reactions`, measure precision/recall with confidence intervals for
`info_present` and `attitude_present`, and gate on precision >= 0.85 and recall >= 0.85 (lower Wilson
CI bound) for both fields. On a miss, iterate the judge prompt/config up to 3 times; if still not
met, escalate with the measured ceiling rather than silently lowering the bar (plan Sec.8 P2 /
Risk #12).

Every reaction is accounted for in the ledger (graded, or recorded as a named failure) -- the same
"no silent drop" discipline the plan states for P5's larger ledger (Sec.8 P5 gate), applied here at
P2's smaller scale so the pattern is established before P5 needs it at volume.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..events import CorpusEntry
from .backend import JudgeABackend, MockJudgeBackend
from .stats import WilsonInterval, wilson_interval

FIELDS = ("info_present", "attitude_present")
PRECISION_RECALL_BAR = 0.85  # provisional floor, plan Sec.8 P2's own justification
MAX_ITERATIONS = 3
MIN_TOTAL_REACTIONS = 100
MIN_PER_DIRECTION = 30


@dataclass(frozen=True)
class LedgerRow:
    event_id: str
    reaction_index: int
    field: str
    ground_truth: bool
    predicted: bool
    correct: bool


@dataclass(frozen=True)
class FieldResult:
    field: str
    tp: int
    fp: int
    tn: int
    fn: int
    precision_ci: WilsonInterval
    recall_ci: WilsonInterval

    @property
    def passes(self) -> bool:
        return self.precision_ci.lower >= PRECISION_RECALL_BAR and self.recall_ci.lower >= PRECISION_RECALL_BAR


@dataclass(frozen=True)
class IterationResult:
    iteration: int
    model_id: str
    ledger: tuple[LedgerRow, ...]
    field_results: dict[str, FieldResult]

    @property
    def all_fields_pass(self) -> bool:
        return all(r.passes for r in self.field_results.values())


@dataclass(frozen=True)
class ValidationOutcome:
    verdict: str  # "PASS" | "ESCALATE"
    iterations: tuple[IterationResult, ...]
    best_iteration: IterationResult
    escalation_message: str = ""


def _collect_reactions(entries: list[CorpusEntry]) -> list[tuple[str, int, "object"]]:
    out = []
    for e in entries:
        for i, vr in enumerate(e.validation_reactions):
            out.append((e.event.id, i, vr))
    return out


def _collect_reactions_with_entry(entries: list[CorpusEntry]) -> list[tuple[str, int, "object", CorpusEntry]]:
    """Same as `_collect_reactions`, plus the owning `CorpusEntry` -- needed only by the real-backend
    path (`answer_info`/`answer_attitude`) to supply the actual `required_info` item text and
    `intended_attitude`, which the mock's ground-truth-cheat path (`_with_truth`) never needed."""

    out = []
    for e in entries:
        for i, vr in enumerate(e.validation_reactions):
            out.append((e.event.id, i, vr, e))
    return out


def check_corpus_sizing(entries: list[CorpusEntry]) -> None:
    """Plan Sec.8 P2's own sizing gate on the validation set, checked before any grading runs."""

    reactions = _collect_reactions(entries)
    if len(reactions) < MIN_TOTAL_REACTIONS:
        raise ValueError(f"validation set has {len(reactions)} reactions, need >= {MIN_TOTAL_REACTIONS}")
    for field in FIELDS:
        n_true = sum(1 for _, _, vr in reactions if getattr(vr, field))
        n_false = len(reactions) - n_true
        if n_true < MIN_PER_DIRECTION or n_false < MIN_PER_DIRECTION:
            raise ValueError(
                f"{field}: {n_true} True / {n_false} False, need >= {MIN_PER_DIRECTION} per direction"
            )


def run_one_iteration(entries: list[CorpusEntry], backend: JudgeABackend, iteration: int) -> IterationResult:
    """Grades every validation reaction and builds the ledger. Dispatches on which interface the
    backend implements:

    - `MockJudgeBackend` (has `answer_info_with_truth`): the existing ground-truth-cheat path,
      unchanged -- proves the precision/recall/CI/escalation machinery independent of any real
      model call.
    - a real `JudgeABackend` (`OllamaJudgeBackend`, `HostedApiJudgeBackend`): calls the abstract
      `answer_info`/`answer_attitude` interface with the reaction's actual text and the owning
      entry's `required_info`/`intended_attitude`, per the interface those methods were already
      specified with (T-2011 finding: the interface existed but nothing before this build ever
      called it with real content -- `answer_info_with_truth`/`answer_attitude_with_truth` are
      mock-only test doubles that were never meant to stand in for a real judge call).
    """

    is_mock = hasattr(backend, "answer_info_with_truth")
    ledger: list[LedgerRow] = []

    if is_mock:
        reactions = _collect_reactions(entries)
        for event_id, idx, vr in reactions:
            pred_info = backend.answer_info_with_truth(vr.info_present)
            ledger.append(
                LedgerRow(
                    event_id=event_id,
                    reaction_index=idx,
                    field="info_present",
                    ground_truth=vr.info_present,
                    predicted=pred_info,
                    correct=pred_info == vr.info_present,
                )
            )
            pred_attitude = backend.answer_attitude_with_truth(vr.attitude_present)
            ledger.append(
                LedgerRow(
                    event_id=event_id,
                    reaction_index=idx,
                    field="attitude_present",
                    ground_truth=vr.attitude_present,
                    predicted=pred_attitude,
                    correct=pred_attitude == vr.attitude_present,
                )
            )
    else:
        reactions_with_entry = _collect_reactions_with_entry(entries)
        for event_id, idx, vr, entry in reactions_with_entry:
            if vr.info_target_index < 0:
                raise ValueError(
                    f"{event_id}[{idx}]: info_target_index={vr.info_target_index} names no single "
                    f"required_info item to ask the real judge about -- this corpus's validation "
                    f"reactions are constructed to always target index 0 (validation_reactions.py); "
                    f"a -1 ('off-topic w.r.t. every item') reaction is a schema case this real-backend "
                    f"path does not have an unambiguous question for and does not guess at."
                )
            info_item = entry.required_info[vr.info_target_index]
            pred_info = backend.answer_info(vr.text, info_item)
            ledger.append(
                LedgerRow(
                    event_id=event_id,
                    reaction_index=idx,
                    field="info_present",
                    ground_truth=vr.info_present,
                    predicted=pred_info,
                    correct=pred_info == vr.info_present,
                )
            )
            pred_attitude = backend.answer_attitude(vr.text, entry.intended_attitude)
            ledger.append(
                LedgerRow(
                    event_id=event_id,
                    reaction_index=idx,
                    field="attitude_present",
                    ground_truth=vr.attitude_present,
                    predicted=pred_attitude,
                    correct=pred_attitude == vr.attitude_present,
                )
            )
        reactions = reactions_with_entry

    field_results: dict[str, FieldResult] = {}
    for f in FIELDS:
        rows = [r for r in ledger if r.field == f]
        tp = sum(1 for r in rows if r.ground_truth and r.predicted)
        fp = sum(1 for r in rows if (not r.ground_truth) and r.predicted)
        tn = sum(1 for r in rows if (not r.ground_truth) and (not r.predicted))
        fn = sum(1 for r in rows if r.ground_truth and (not r.predicted))
        predicted_positive = tp + fp
        actual_positive = tp + fn
        precision_ci = wilson_interval(tp, predicted_positive) if predicted_positive > 0 else wilson_interval(0, 1)
        recall_ci = wilson_interval(tp, actual_positive) if actual_positive > 0 else wilson_interval(0, 1)
        field_results[f] = FieldResult(field=f, tp=tp, fp=fp, tn=tn, fn=fn, precision_ci=precision_ci, recall_ci=recall_ci)

    n_reactions = len(reactions)
    if len(ledger) != n_reactions * len(FIELDS):
        raise AssertionError(
            f"ledger accounts for {len(ledger)} cells, expected {n_reactions * len(FIELDS)} "
            f"({n_reactions} reactions x {len(FIELDS)} fields) -- a silent drop occurred"
        )

    return IterationResult(iteration=iteration, model_id=backend.model_id, ledger=tuple(ledger), field_results=field_results)


def run_validation(entries: list[CorpusEntry], backend_factory) -> ValidationOutcome:
    """`backend_factory(iteration: int) -> JudgeABackend` builds the judge (mock or real) to try for
    a given iteration (1-indexed) -- e.g. a fresh `OllamaJudgeBackend` with a reworded prompt on a
    retry. Iterates up to MAX_ITERATIONS times; stops at the first iteration where every field
    passes. If none pass, returns verdict "ESCALATE" naming the best (highest-minimum-lower-CI-bound)
    iteration's numbers as the measured ceiling -- plan Sec.8 P2 / Risk #12: escalate to Dan with the
    measured ceiling rather than lowering the bar silently."""

    check_corpus_sizing(entries)

    iterations: list[IterationResult] = []
    for i in range(1, MAX_ITERATIONS + 1):
        backend = backend_factory(i)
        result = run_one_iteration(entries, backend, iteration=i)
        iterations.append(result)
        if result.all_fields_pass:
            return ValidationOutcome(verdict="PASS", iterations=tuple(iterations), best_iteration=result)

    def _worst_lower_bound(r: IterationResult) -> float:
        return min(min(fr.precision_ci.lower, fr.recall_ci.lower) for fr in r.field_results.values())

    best = max(iterations, key=_worst_lower_bound)
    ceiling_desc = "; ".join(
        f"{f}: precision {fr.precision_ci.point_estimate:.3f} (CI lower {fr.precision_ci.lower:.3f}), "
        f"recall {fr.recall_ci.point_estimate:.3f} (CI lower {fr.recall_ci.lower:.3f})"
        for f, fr in best.field_results.items()
    )
    message = (
        f"Judge A did not clear precision>=0.85 and recall>=0.85 (lower CI bound) on both fields "
        f"after {len(iterations)} iterations. Measured ceiling (iteration {best.iteration}, "
        f"model {best.model_id}): {ceiling_desc}. Escalating to Dan per plan Sec.8 P2 / Risk #12 -- "
        f"options are narrowing the judge's task further (one field per call) or adding a second "
        f"cross-check judge for disagreement arbitration."
    )
    return ValidationOutcome(verdict="ESCALATE", iterations=tuple(iterations), best_iteration=best, escalation_message=message)
