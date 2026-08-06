#!/usr/bin/env python3
"""§12 criterion 2's producer -- the Python reference side of the trace-for-trace
comparison (T-1522, SuperSLM_S3a_WalkingSkeleton_Plan.md Sec11 S3.1c item 3).

Invokes `superslm_spike.dynamic_engine.forward_dynamic_vec` -- the vectorized
engine, bit-equal to the scalar `pipeline.forward_dynamic` by the vendored
closure's own committed suite, and the one the plan names because the scalar
measures ~4.7 min/token at the real checkpoint while criterion 2 compares
every trace record for every reference prompt -- once per member of the
closed five-member reference pack (D-SLM350), collecting the chain records
(`_chain_record_vec`) and the K/V-landing records the reference already
builds, plus the final logits, and serializes the stream to JSON Lines, one
record per line, in the field order `trace_hook.h` pins:

    chain record:      site, token_index, x_int, Dprime, Dn, s, R, codes, m_out, e_out
    K/V-landing record: site, token_index, head, x_int, m_in, e_in, codes, m_out, e_out

`dynamic_engine`'s own trace dicts are already built in exactly this key
order (confirmed at source, `dynamic_engine.py`'s `_chain_record_vec` and its
K/V-landing `trace.append` calls) -- this script's serialization step
reproduces that order explicitly rather than relying on it implicitly, so a
future reordering in the engine's own trace-building code is not silently
inherited as a wire-format change.

A third record kind, `final_logits`, closes each member's stream (the plan's
own "plus the final logits" -- its shape is not one `trace_hook.h` pins,
since that schema covers per-site emission only): `{"kind": "final_logits",
"member_index": <int>, "logits": [<int>, ...]}`. The consumer/comparator
(Stage 4, T-1522's other half, outside this build's scope) reads this stream;
this script's own contract for it is the JSON_SCHEMA_NOTE below.

WHAT THIS SCRIPT DOES NOT DO. It does not load a real checkpoint itself, does
not calibrate, and does not touch `transformers` or a tokenizer -- both would
pull real-weight-adjacent, cross-repo dependencies into the CI-facing path
the vendoring boundary (Sec11 S3.1c item 1) exists to keep out. It reads:

  - the reference pack's token ids from a PRECOMPUTED, PINNED fixture
    (`superslm_spike/criterion2_prompt_pack_pinned.json`, produced once by
    `precompute_criterion2_prompt_pack.py` against the real tokenizer and
    committed -- this script never renders a prompt itself);
  - a `model` object (a `superslm_spike.pipeline.QuantizedModel`) from a
    Python pickle at `--model`, produced however the caller likes (a fixture
    model for a hermetic test, or a real calibrated artifact for a genuine
    end-to-end run) -- this script does not construct one itself, so it
    carries no dependency on how a real artifact is loaded or where it lives.

JSON_SCHEMA_NOTE: every record is a JSON object on its own line (JSON Lines,
not a single JSON array), so a consumer can stream the file without holding
the whole run in memory -- the same shape `trace_hook.h`'s own record stream
takes on the C++ side.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)

DEFAULT_PROMPTS_PATH = os.path.join(_THIS_DIR, "superslm_spike", "criterion2_prompt_pack_pinned.json")

# The pinned field order (trace_hook.h; SuperSLM_S3a_WalkingSkeleton_Plan.md
# Sec11 S3.1c item 3). dynamic_engine's own trace dicts already carry these
# keys in this order; this tuple is what makes that order an explicit,
# checked contract of THIS script rather than an inherited accident.
_CHAIN_FIELDS = ("site", "token_index", "x_int", "Dprime", "Dn", "s", "R", "codes", "m_out", "e_out")
_KV_FIELDS = ("site", "token_index", "head", "x_int", "m_in", "e_in", "codes", "m_out", "e_out")


def _jsonable(value):
    """Recursively converts a dynamic_engine trace value (or the forward's
    own logits array) to a JSON-safe Python primitive: tuples/lists recurse
    element-wise; a numpy array (the logits, `int32`, one row per token)
    converts via `.tolist()` -- NOT `.item()`, which only accepts a
    size-1 array and would raise on the logits' real shape; a numpy SCALAR
    (e.g. a single `np.int64` inside a trace record) converts via `.item()`;
    everything else (a plain `str`/`int`/`float`) passes through unchanged."""
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        # numpy scalars and arrays both expose .tolist()/.item(); .tolist()
        # handles both shapes uniformly (a 0-d/scalar array's .tolist() is
        # the same plain Python number .item() would return).
        return value.tolist()
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        return value.item()
    return value


def _order_record(record: dict) -> dict:
    """Re-keys one dynamic_engine trace dict into the pinned field order,
    converting every value to a JSON-safe primitive. Raises KeyError naming
    the record and the missing field if dynamic_engine ever stops emitting
    one of the pinned fields -- a silent partial record is exactly the shape
    a real forward defect could take, and this script never emits one."""
    fields = _KV_FIELDS if "head" in record else _CHAIN_FIELDS
    ordered = {}
    for field in fields:
        if field not in record:
            raise KeyError(
                f"trace record for site={record.get('site')!r} token_index="
                f"{record.get('token_index')!r} is missing pinned field {field!r} "
                f"-- dynamic_engine's own record shape has drifted from the pin"
            )
        ordered[field] = _jsonable(record[field])
    return ordered


def load_prompt_pack(prompts_path: str) -> list[dict]:
    """Reads the precomputed, pinned reference-pack fixture -- a list of
    {index, axis, source, expected_verdict, input_ids} dicts, one per member,
    in index order. Raises FileNotFoundError naming the path (with a pointer
    to the precompute script) if the fixture has not been generated."""
    if not os.path.isfile(prompts_path):
        raise FileNotFoundError(
            f"{prompts_path}: no pinned reference pack found. Run "
            f"tests/reference/precompute_criterion2_prompt_pack.py once (with "
            f"SUPERSLM_REFERENCE_SOURCE pointing at a D:\\Wizard checkout) to "
            f"produce it before running the producer."
        )
    with open(prompts_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    members = sorted(payload["members"], key=lambda m: m["index"])
    return members


def load_model(model_path: str):
    """Unpickles a `superslm_spike.pipeline.QuantizedModel` from `model_path`.
    `superslm_spike` must already be importable (sys.path is set up by the
    caller) so the pickled object's class resolves to the vendored copy.

    The pickle need not carry a live `float_source` or `tokenize_prompt` --
    `forward_dynamic_vec` touches neither (both are calibration/prompt-
    rendering concerns, not forward concerns), and neither is reliably
    picklable in general: `float_source` is a closure over a live checkpoint
    reader (or a calibration-time float array), and `tokenize_prompt` is a
    closure over a live tokenizer. A caller loading a real artifact via
    `artifact_cache.load_artifact` (not part of this repository's vendored
    closure; a cross-repo, hand-run concern, mirroring
    `precompute_criterion2_prompt_pack.py`'s own convention) replaces both
    with `None` via `dataclasses.replace` before pickling, since
    `QuantizedModel` is a frozen dataclass."""
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"{model_path}: no model pickle found")
    with open(model_path, "rb") as f:
        return pickle.load(f)


def run_member(model, member: dict) -> tuple[list[dict], list[int]]:
    """Runs one reference-pack member through `forward_dynamic_vec`, returning
    (ordered trace records, final logits as a plain list of ints)."""
    from superslm_spike import dynamic_engine

    trace: list = []
    logits = dynamic_engine.forward_dynamic_vec(model, member["input_ids"], trace=trace)
    ordered_records = [_order_record(r) for r in trace]
    logits_list = _jsonable(logits)
    if logits_list and isinstance(logits_list[0], list):
        # int32 logits is one row per token; the trace record contract only
        # pins per-site records, so this script does not further validate
        # shape beyond making it JSON-safe -- the comparator (Stage 4, out of
        # this build's scope) is the consumer that interprets it.
        pass
    return ordered_records, logits_list


def run_pack(model, members: list[dict], out) -> int:
    """Runs every member in order, writing one JSON object per line to `out`
    (a writable file object) -- the per-site records first, then a
    `final_logits` record closing that member's stream. Returns the total
    record count written (site records across all members; `final_logits`
    lines are counted separately by the caller if needed)."""
    total = 0
    for member in members:
        records, logits = run_member(model, member)
        for record in records:
            out.write(json.dumps(record, separators=(",", ":")) + "\n")
            total += 1
        out.write(
            json.dumps(
                {"kind": "final_logits", "member_index": member["index"], "logits": logits},
                separators=(",", ":"),
            )
            + "\n"
        )
    return total


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="path to a pickled superslm_spike.pipeline.QuantizedModel")
    parser.add_argument("--prompts", default=DEFAULT_PROMPTS_PATH, help="path to the pinned reference-pack JSON")
    parser.add_argument("--out", required=True, help="output path for the JSON Lines trace stream")
    args = parser.parse_args(argv)

    try:
        members = load_prompt_pack(args.prompts)
        model = load_model(args.model)
    except FileNotFoundError as e:
        print(f"run_criterion2_trace.py: FAILED -- {e}", file=sys.stderr)
        return 1

    with open(args.out, "w", encoding="utf-8", newline="\n") as out:
        record_count = run_pack(model, members, out)

    print(
        f"run_criterion2_trace.py: OK -- {len(members)} reference-pack member(s), "
        f"{record_count} site record(s) written to {args.out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
